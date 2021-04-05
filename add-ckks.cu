//
// Created by ahmetcanmert on 14.02.2021.
//

#define PROFILE

#include "palisade.h"
#include "palisadecore.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdlib.h>
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


/*
void copyDCRTPolyD2H(DCRTPoly pol, unsigned long long *dev) {
    int q_size = pol.GetParams()->GetParams().size();
    vector<PolyImpl<NativeVector>> rns_poly = pol.GetAllElements();
    int n = rns_poly[0].GetRingDimension();
    unsigned long long * poly_h = (unsigned long long*)malloc(n*q_size*sizeof(unsigned long long));
    cudaMemcpy(poly_h, dev, sizeof(unsigned long long) * n * q_size,cudaMemcpyDeviceToHost);

//    for (int k = 0; k < q_size; k++, index+=n){
//
//        std::vector<unsigned long long> v(poly_h, poly_h + n);
//        rns_poly[k] = v;
//    }
    std::vector<int64_t> v(poly_h, poly_h + (n*q_size));
    pol = v;
    std::cout << "Operation Completed" << std::endl;
}
*/



void copyDCRTPolyH2D(unsigned long long *&dest, DCRTPoly src) {

    unsigned long long index = 0;
    int q_size = src.GetParams()->GetParams().size();
    vector<PolyImpl<NativeVector>> rns_poly = src.GetAllElements();
    int n = rns_poly[0].GetRingDimension();
    for (int k = 0; k < q_size; k++, index+=n)
        cudaMemcpy(dest+index, &rns_poly[k].GetValues()[0], sizeof(unsigned long long) * n,cudaMemcpyHostToDevice);
}
// Kernel
__global__ void add1_poly(unsigned long long *des, unsigned long long *src)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    des[id] = src[id] + 1;
}


// Kernel for poyl addition
__global__ void add_poly(long int *d_poly_a, long int *d_poly_b, long int *c, int d_N, int d_q_modulus)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	for (int k=0; k<d_N*2; k++)
	{
		printf("%lu\n", d_poly_a[k]);
		c[k] = (d_poly_a[k] + d_poly_b[k]) % d_q_modulus;

	}
//	printf("%d\n", c[id]);

}

// Kernel for poly multiplication
// assuming <<2, 16>> for testing
__global__ void mult_poly(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned int N, unsigned int q_modulus)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int idx = blockIdx.x;

	if (idx == 0) {
		c[id] = (a[id] * b[id]) % q_modulus;
		c[id+N] =  (a[id+N] * b[id+N]) % q_modulus;

	}
	
	if (idx == 1) {
		c[id+N] = (a[id%N] * b[id]) % q_modulus;
		c[id+2*N] = (a[id] * b[id%N]) % q_modulus;

	}

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
    cudaMalloc(&sk_device, sizeof(unsigned long long) * size * n);
    cudaMalloc(&sk_plus_1_device, sizeof(unsigned long long) * size * n);
    TIC(t);
    copyDCRTPolyH2D(sk_device, keys.secretKey->GetPrivateElement());
    ioH2D = TOC_US(t);
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(n) / thr_per_blk );

    // Launch kernel
    add1_poly<<< blk_in_grid, thr_per_blk >>>(sk_plus_1_device, sk_device);
    DCRTPoly sk_plus_1 = keys.secretKey->GetPrivateElement();
    TIC(t);

    //copyDCRTPolyD2H(sk_plus_1, sk_plus_1_device);
    ioD2H = TOC_US(t);

    std::cout << "Host to Device IO Timing: " << ioH2D << " micros" << std::endl;
    std::cout << "Device to Host IO Timing: " << ioD2H << " micros" << std::endl;


    // Adding two polynomials
    int q_modulus = 78230497;
    int N = 16;

    long int poly_a[] = {25541959, 33438588, 60725242, 60529208, 42397239, 22224914, 2124676, 34198137, 71507853, 71824477, 61357045, 72845178, 17736451, 29457145, 45762325, 45950525,1714289, 37737577, 54276863, 13183248, 75898501, 40027669, 76953410, 11736972, 16473303, 13235781, 75482361, 17117934, 74884319, 16464100, 65816274, 60527152};
    long int poly_b[] = {59482001, 67025803, 379920, 63444484, 48179845, 53488173, 73999829, 68947505, 16841640, 52220750, 56859592, 8241134, 18184706, 46715793, 4658958, 58141010, 11673823, 58609589, 19832435, 68579871, 73899106, 36726229, 55000108, 1773173, 24848758, 26299516, 4951156, 28272420, 36678120, 62696297, 43312646, 43399602};

    long int *a = (long int*)malloc(sizeof(long int)* 2*N);
    long int *b = (long int*)malloc(sizeof(long int)* 2*N);


    for (int j=0; j<2*N; j++)
    {

	a[j] = poly_a[j];
	b[j] = poly_b[j];

    }    
    
    printf("%lu ", a[0]);
    printf("%lu ", b[0]);



    long int *d_poly_a, *d_poly_b, *d_poly_c;
    //unsigned int d_q_modulus;	
    //unsigned int d_n;

    cudaMalloc((void**) &d_poly_a, sizeof(long int) * 2 * N);
    cudaMalloc((void**) &d_poly_b, sizeof(long int) * 2 * N);
    cudaMalloc((void**) &d_poly_c, sizeof(long int) * 2 * N);

    //cudaMalloc(&d_q_modulus, sizeof(unsigned int));
    //cudaMalloc(&d_n, sizeof(unsigned int));

    cudaMemcpy(a, d_poly_a, sizeof(long int) * 2 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b, d_poly_b, sizeof(long int) * 2 * N, cudaMemcpyHostToDevice);

    //cudaMemcpy(q_modulus, d_q_modulus, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(n, d_n, sizeof( int), cudaMemcpyHostToDevice);

    add_poly<<<1, 1>>>(d_poly_a, d_poly_b, d_poly_c, N, q_modulus);    
    cudaDeviceSynchronize();

    std::cout << "Adding two polynomials is finished, time to check correctness!!" << std::endl;

    long int *res_poly = (long int*)malloc(N*2*sizeof(long int));
    cudaMemcpy(res_poly, d_poly_c, sizeof(long int)* 2*N, cudaMemcpyDeviceToHost);

    std::cout << "Copying the result poly back from the device!!" << std::endl;    

    for (int i=0; i< 2*N; i++){

	std::cout << res_poly[i] << std::endl;
	//std::cout << poly_a[i] << std::endl;
	//std::cout << poly_b[i] << std::endl;
    }

    std::cout<< "Printing the result has ended" << std::endl; 
    
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
