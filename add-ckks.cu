//
// Created by ahmetcanmert on 14.02.2021.
//

#define PROFILE

#include "palisade.h"

using namespace lbcrypto;

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void copyDCRTPolyH2D(unsigned long long *&dest, DCRTPoly src) {

    unsigned long long index = 0;
    int q_size = src.GetParams()->GetParams().size();
    vector<PolyImpl<NativeVector>> rns_poly = src.GetAllElements();
    int n = rns_poly[0].GetRingDimension();
    for (int k = 0; k < q_size; k++, index+=n)
        CUDA_CALL(cudaMemcpy(dest+index, &rns_poly[k].GetValues()[0], sizeof(unsigned long long) * n,cudaMemcpyHostToDevice));
}
// Kernel
__global__ void add1_poly(unsigned long long *des, unsigned long long *src)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    des[id] = src[id] + 1;
}

void copyDCRTPolyD2H(DCRTPoly poly, unsigned long long *dev) {
    int q_size = poly.GetParams()->GetParams().size();
    vector<PolyImpl<NativeVector>> rns_poly = poly.GetAllElements();
    int n = rns_poly[0].GetRingDimension();
    unsigned long long * poly_h = (unsigned long long*)malloc(n*q_size*sizeof(unsigned long long));
    CUDA_CALL(cudaMemcpy(poly_h, dev, sizeof(unsigned long long) * n * q_size,cudaMemcpyDeviceToHost));

//    for (int k = 0; k < q_size; k++, index+=n){
//
//        std::vector<unsigned long long> v(poly_h, poly_h + n);
//        rns_poly[k] = v;
//    }
    std::vector<int64_t> v(poly_h, poly_h + (n*q_size));
    poly = v;
    std::cout << "Operation Completed" << std::endl;
}

int main() {

    TimeVar t;

    double ioH2D(0.0);
    double ioD2H(0.0);

    uint32_t multDepth = 1;
    uint32_t scaleFactorBits = 50;

    uint32_t batchSize = 8;
    SecurityLevel securityLevel = HEStd_128_classic;

    // The following call creates a CKKS crypto context based on the
    // arguments defined above.
    CryptoContext<DCRTPoly> cc =
            CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
                    multDepth, scaleFactorBits, batchSize, securityLevel);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension()
              << std::endl
              << std::endl;
    int n = cc->GetRingDimension();
    int size = cc->GetCryptoParameters()->GetElementParams()->GetParams().size();

    // Enable the features that you wish to use
    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);


    // B. Step 2: Key Generation
    auto keys = cc->KeyGen();

    cc->EvalMultKeyGen(keys.secretKey);

    cc->EvalAtIndexKeyGen(keys.secretKey, {1, -2});
    // TIC(t);
    // Create GPU DRCTPoly
    unsigned long long* sk_device, *sk_plus_1_device;
    CUDA_CALL(cudaMalloc(&sk_device, sizeof(unsigned long long) * size * n));
    CUDA_CALL(cudaMalloc(&sk_plus_1_device, sizeof(unsigned long long) * size * n));
    TIC(t);
    copyDCRTPolyH2D(sk_device, keys.secretKey->GetPrivateElement());
    ioH2D = TOC_US(t);
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(n) / thr_per_blk );

    // Launch kernel
    add1_poly<<< blk_in_grid, thr_per_blk >>>(sk_plus_1_device, sk_device);
    DCRTPoly sk_plus_1 = keys.secretKey->GetPrivateElement();
    TIC(t);
    copyDCRTPolyD2H(sk_plus_1,sk_plus_1_device);
    ioD2H = TOC_US(t);

    std::cout << "Host to Device IO Timing: " << ioH2D << " micros" << std::endl;
    std::cout << "Device to Host IO Timing: " << ioD2H << " micros" << std::endl;

    // Step 3: Encoding and encryption of inputs

//    // Inputs
//    vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
//    vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};
//
//    // Encoding as plaintexts
//    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
//    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
//
//    std::cout << "Input x1: " << ptxt1 << std::endl;
//    std::cout << "Input x2: " << ptxt2 << std::endl;
//
//    // Encrypt the encoded vectors
//    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
//    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
//
//    // Create GPU DRCTPoly
//    unsigned long long* c1_device;
//    CUDA_CALL(cudaMalloc(&c1_device, sizeof(unsigned long long) * size * n));
//
//    // Step 4: Evaluation
//
//    // Homomorphic addition
//    auto cAdd = cc->EvalAdd(c1, c2);
//
//
//    // Step 5: Decryption and output
//    Plaintext result;
//    // We set the cout precision to 8 decimal digits for a nicer output.
//    // If you want to see the error/noise introduced by CKKS, bump it up
//    // to 15 and it should become visible.
//    std::cout.precision(8);
//    std::cout << std::endl
//              << "Results of homomorphic computations: " << std::endl;
//
//    // Decrypt the result of addition
//    cc->Decrypt(keys.secretKey, cAdd, &result);
//    result->SetLength(batchSize);
//    std::cout << "x1 + x2 = " << result;
//    std::cout << "Estimated precision in bits: " << result->GetLogPrecision()
//              << std::endl;


    return 0;
}
