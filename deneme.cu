#include <iostream>

#include "cryptocontexthelper.h"
#include "subgaussian/subgaussian.h"
#include <errno.h>

//#include "palisade.h"
//#include "palisadecore.h"

//#include "cryptocontexthelper.h"
//#include "/usr/local/include/palisade/trapdoor/abe/kp_abe_rns.h"

#include "abe/kp_abe_rns.h"

using namespace std;
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



void copyRow(unsigned long long * &A, unsigned long long * &B, int m, unsigned q_amount, unsigned n) {
    unsigned int index = 0;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < q_amount; j++, index+=n){
            // DELETING THE CUDA CALL
		cudaMemcpy(A+index, B+index, sizeof(unsigned long long) * n, cudaMemcpyDeviceToDevice);
        }
    }
}


int KPABE_BenchmarkCircuitTestDCRT(usint iter, int32_t base, usint n, size_t size, usint ell);
usint EvalNANDTree(usint *x, usint ell);


__global__ void NAND_circuit_1(int i, int m_m ,unsigned long long* &wCT, unsigned long long * origCT_device, unsigned long long *&wPublicElementB, unsigned long long * p_psi, unsigned long long *&negPubElemB_device){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < m_m){
//		printf("\%llu \n", origCT_device[idx]);
		wCT[i + idx] = origCT_device[(2*i + 1) *m_m + idx];	

	}
}


int main() {
//    PalisadeParallelControls.Enable();

    cout << "BEFORE CALLING THE KPABE BENCH" << endl;
    //PseudoRandomNumberGenerator::InitPRNG();
    usint  iter = 4;

    usint  att = 2;
    usint q_size = 4;
    usint n =  1 << 12;
    usint  base = 1 << 20;

    KPABE_BenchmarkCircuitTestDCRT(iter, base,n,q_size,att);

    return 0;
}

void copyMatrixH2D2(unsigned long long *&tmp, Matrix<DCRTPoly> src) {
    vector<vector<DCRTPoly>> matrix = src.GetData();
    unsigned long long index = 0;
    for (int i = 0; i < src.GetRows(); i++){
        for (int j = 0; j < src.GetRows(); j++){
            int q_size = matrix[i][j].GetParams()->GetParams().size();
            vector<PolyImpl<NativeVector>> rns_poly = matrix[i][j].GetAllElements();
            int n = rns_poly[0].GetRingDimension();
            for (int k = 0; k < q_size; k++, index+=n){
                //CUDA_CALL(cudaMemcpy(dest+index, &rns_poly[k].GetValues()[0], sizeof(unsigned long long) * n,cudaMemcpyHostToDevice));
                cudaMemcpy(&tmp+index, &rns_poly[k].GetValues()[0], sizeof(unsigned long long) * n, cudaMemcpyHostToHost);
            }
	}
    }
}

void copyMatrixH2D(unsigned long long *&dest, Matrix<DCRTPoly> src) {
    vector<vector<DCRTPoly>> matrix = src.GetData();
    unsigned long long index = 0;
    for (int i = 0; i < src.GetRows(); i++){
        for (int j = 0; j < src.GetRows(); j++){
            int q_size = matrix[i][j].GetParams()->GetParams().size();
            vector<PolyImpl<NativeVector>> rns_poly = matrix[i][j].GetAllElements();
            int n = rns_poly[0].GetRingDimension();
            for (int k = 0; k < q_size; k++, index+=n)
                //CUDA_CALL(cudaMemcpy(dest+index, &rns_poly[k].GetValues()[0], sizeof(unsigned long long) * n,cudaMemcpyHostToDevice));
		cudaMemcpy(dest+index, &rns_poly[k].GetValues()[0], sizeof(unsigned long long) * n,cudaMemcpyHostToDevice);
        }
    }
}


void print_array(unsigned long long a[])
{
    cout << "[";
    for (int i = 0; i < 1024; i++)
    {
        cout << a[i] << ", ";
    }
    cout << "]\n";
}
void printMatrix(unsigned long long int *d_matrix, Matrix<DCRTPoly> h_matrix, unsigned i, unsigned m, unsigned size,
                 unsigned n) {
    vector<vector<DCRTPoly>> matrix = h_matrix.GetData();
    unsigned long long index = 0;
    for (int i = 0; i < h_matrix.GetRows(); i++){
        for (int j = 0; j < h_matrix.GetRows(); j++){
            int q_size = matrix[i][j].GetParams()->GetParams().size();
            vector<PolyImpl<NativeVector>> rns_poly = matrix[i][j].GetAllElements();
            int n = rns_poly[0].GetRingDimension();
            for (int k = 0; k < q_size; k++, index+=n)
                print_array(d_matrix+index);
        }
    }

}
void EvalCT_GPU(KPABErns &kpabe, const shared_ptr<ILDCRTParams<BigInteger>> &params, Matrix<DCRTPoly> &pubElemB, unsigned long long* &negPubElemB_device,
                usint x[], usint *x_device, unsigned long long* &origCT_device, Matrix<DCRTPoly> &origCT, usint *&evalAttributes_dev,
                unsigned long long* &evalCT_device, usint ell, usint M, size_t size) {

    vector<LatticeSubgaussianUtility<NativeInteger>> m_util = kpabe.get_util();
    //usint ell = kpabe.Get_ell();

    usint m = M; //params->GetModulus().GetMSB()+2 ; //kpabe.Get_m();
    usint n = params->GetRingDimension();
    usint q_size = params->GetParams().size();

    //cout << "m: "<< m<< endl;
    auto zero_alloc = DCRTPoly::Allocator(params, Format::EVALUATION);

    // Part pertaining to A (does not change)
    //cout << "START OF COPY_ROW FUNCTION" << endl;
    copyRow(evalCT_device, origCT_device, m, q_size, n);
    usint gateCnt = ell - 1;
    
    //cout << "AFTER COPYING ROW" << endl;
    // Matrix<DCRTPoly> psi(zero_alloc, m_m, m_m);
    // w stands for Wire
    unsigned long long* wPublicElementB; 
//    createMatrix(wPublicElementB, gateCnt, m, q_size, n);  // Bis associated with internal wires of the circuit
    cudaMalloc(&wPublicElementB, sizeof(unsigned long long) * gateCnt * m * q_size * n);

    //cout << "AFTER FIRST CUDA CALL" << endl;
    unsigned long long* wCT;
//    createMatrix(wCT, gateCnt, m, q_size, n); // Ciphertexts associated with internal wires of the circuit
    cudaMalloc(&wCT, sizeof(unsigned long long) * gateCnt * m * q_size * n);

    // Attribute values associated with internal wires of the circuit
    //TODO check this one
    std::vector<usint> wX(gateCnt);

    // Temporary variables for bit decomposition operation
    Matrix<DCRTPoly> negB(zero_alloc, 1, m);  // Format::EVALUATION (NTT domain)
//#/    unsigned long long* negB;
//    createMatrix(negB, gateCnt, m, q_size, n); // Format::EVALUATION (NTT domain)
//#/    CUDA_CALL(cudaMalloc(&negB, sizeof(unsigned long long) * gateCnt * m * q_size * n));
    // Input level of the circuit
    usint t = ell >> 1;  // the number of the gates in the first level (the
    // number of input gates)

    unsigned long long* p_psi;

    // looping to evaluate and calculate w, wB, wC
    // and R for all first level input gates
    for (usint i = 0; i < t; i++){
        wX[i] = x[0] - x[2 * i + 1] * x[2 * i + 2];  // calculating binary wire value
//
#pragma omp parallel for schedule(dynamic)
    for (usint j = 0; j < m; j++) {  // Negating Bis for bit decomposition
        negB(0, j) = pubElemB(2 * i + 1, j).Negate();
        negB(0, j).SwitchFormat();
    }

    auto psi = InverseRingVectorDCRT(m_util, negB, 1);
    psi->SwitchFormat();

    cudaMalloc(reinterpret_cast<void **>(&p_psi), sizeof(unsigned long long) * (ell + 1) * m * size * n);
    copyMatrixH2D(p_psi, (*psi));


    // Newly Added, copying origCT before calling the kernel
    unsigned long long * origCT_device_n;
    cudaMalloc(&origCT_device_n, sizeof(unsigned long long) * (1) * m * size * n);
    copyMatrixH2D(origCT_device_n, origCT.ExtractRows(1, ell+1));


    // Calling the first NAND circuit kernel ## TESTING PHASE
    if (x[2 * i + 2] != 0) {
	    NAND_circuit_1<<<1, m>>>(i, m , wCT, origCT_device_n, wPublicElementB, p_psi, negPubElemB_device);

    unsigned long long * res_wCT;
    cudaMemcpy(res_wCT, &wCT, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost);

/*
    unsigned long long *tmp;
    cudaMallocHost(&tmp, sizeof(unsigned long long) * gateCnt * m * q_size * n);
    copyMatrixH2D2(tmp, origCT);	
*/

    for(int j =0; j< 10; j++){
	
	printf("%llu \n", res_wCT[j]);
    }

}

printf("RESULTS FROM origCT\n");

	Matrix<DCRTPoly> wCT1(zero_alloc, gateCnt, m);
#pragma omp parallel for schedule(dynamic)
/*
    for (usint j = 0; j < m; j++) {
	if (x[2 * i + 2] != 0)
		wCT1(i, j) = origCT(2 * i + 1, j);
        else
                wCT1(i, j).SetValuesToZero();
    }    
*/

    for(int j =0; j< 10; j++){

        printf("%u \n", wCT1(i, j));
//	printf("%u \n", origCT(0, j));
    }

  }


}

int KPABE_BenchmarkCircuitTestDCRT(usint iter, int32_t base, usint n, size_t size, usint ell) { 
// usint n = 1 << 12; // cyclotomic order
    size_t kRes = 50;   // CRT modulus size
//  usint ell = 4;      // No of attributes
//  size_t size = 2;    // Number of CRT moduli

    std::cout << "Number of attributes: " << ell << std::endl;

    std::cout << "n: " << n << std::endl;

    // double sigma = SIGMA;

    std::vector<NativeInteger> moduli;
    std::vector<NativeInteger> roots_Of_Unity;

    // makes sure the first integer is less than 2^60-1 to take advangate of NTL
    // optimizations
    NativeInteger firstInteger = FirstPrime<NativeInteger>(kRes, 2 * n);

    NativeInteger q = PreviousPrime<NativeInteger>(firstInteger, 2 * n);
    moduli.push_back(q);
    roots_Of_Unity.push_back(RootOfUnity<NativeInteger>(2 * n, moduli[0]));
    std::cout << "q["<< 0 <<"]_k: " << q.GetMSB() << std::endl;
    NativeInteger prevQ = q;
    for (size_t i = 1; i < size; i++) {
        prevQ = lbcrypto::PreviousPrime<NativeInteger>(prevQ, 2 * n);
        NativeInteger nextRootOfUnity(RootOfUnity<NativeInteger>(2 * n, prevQ));
        moduli.push_back(prevQ);
        std::cout << "q["<< i <<"]_k: " << moduli[i].GetMSB() << std::endl;
        roots_Of_Unity.push_back(nextRootOfUnity);
    }

    auto ilDCRTParams =
            std::make_shared<ILDCRTParams<BigInteger>>(2 * n, moduli, roots_Of_Unity);

    ChineseRemainderTransformFTT<NativeVector>::PreCompute(roots_Of_Unity, 2 * n,
                                                           moduli);

    std::cout << "k: " << ilDCRTParams->GetModulus().GetMSB() << std::endl;

    size_t digitCount = (long)ceil(
            log2(ilDCRTParams->GetParams()[0]->GetModulus().ConvertToDouble()) /
            log2(base));
    size_t k = digitCount * ilDCRTParams->GetParams().size();

    std::cout << "digit count = " << digitCount << std::endl;
//  std::cout << "k = " << k << std::endl;

    size_t m = k + 2;

    auto zero_alloc = DCRTPoly::Allocator(ilDCRTParams, Format::COEFFICIENT);

    DCRTPoly::DggType dgg = DCRTPoly::DggType(SIGMA);
    DCRTPoly::DugType dug = DCRTPoly::DugType();
    DCRTPoly::BugType bug = DCRTPoly::BugType();

    // Trapdoor Generation
    std::pair<Matrix<DCRTPoly>, RLWETrapdoorPair<DCRTPoly>> trapdoorA =
            RLWETrapdoorUtility<DCRTPoly>::TrapdoorGen(
                    ilDCRTParams, SIGMA, base);  // A.first is the public element

    DCRTPoly pubElemBeta(dug, ilDCRTParams, Format::EVALUATION);

    Matrix<DCRTPoly> publicElementB(zero_alloc, ell + 1, m);
    Matrix<DCRTPoly> ctCin(zero_alloc, ell + 2, m);
    DCRTPoly c1(dug, ilDCRTParams, Format::EVALUATION);

    KPABErns pkg, sender, receiver;

    pkg.Setup(ilDCRTParams, base, ell, dug, &publicElementB);
    sender.Setup(ilDCRTParams, base, ell);
    receiver.Setup(ilDCRTParams, base, ell);

    // Attribute values all are set to 1 for NAND gate Format::EVALUATION
    std::vector<usint> x(ell + 1);
    x[0] = 1;

    usint found = 0;
    while (found == 0) {
        for (usint i = 1; i < ell + 1; i++)
            // x[i] = rand() & 0x1;
            x[i] = bug.GenerateInteger().ConvertToInt();
        if (EvalNANDTree(&x[1], ell) == 0) found = 1;
    }

    usint y;

    TimeVar t1;
    double avg_keygen(0.0), avg_evalct(0.0), avg_evalpk(0.0), avg_enc(0.0),
            avg_dec(0.0);

    // plaintext
    for (usint i = 0; i < iter; i++) {
        std::cout << "running iter " << i + 1 << std::endl;

        NativePoly ptext(bug, ilDCRTParams->GetParams()[0], Format::COEFFICIENT);

        // circuit outputs
        Matrix<DCRTPoly> evalBf(
                DCRTPoly::Allocator(ilDCRTParams, Format::EVALUATION), 1,
                m);  // evaluated Bs
        Matrix<DCRTPoly> evalCf(
                DCRTPoly::Allocator(ilDCRTParams, Format::EVALUATION), 1,
                m);  // evaluated Cs
        Matrix<DCRTPoly> ctCA(DCRTPoly::Allocator(ilDCRTParams, Format::EVALUATION),
                              1, m);  // CA

        // secret key corresponding to the circuit output
        Matrix<DCRTPoly> sk(zero_alloc, 2, m);

        // decrypted text
        NativePoly dtext;


        // Switches to Format::EVALUATION representation
        // ptext.SwitchFormat();
	//cout << "START OF ENCRYPTION " << endl;
        
	TIC(t1);
        sender.Encrypt(ilDCRTParams, trapdoorA.first, publicElementB, pubElemBeta,
                       &x[0], ptext, dgg, dug, bug, &ctCin,
                       &c1);  // Cin and c1 are the ciphertext
        avg_enc += TOC(t1);

        ctCA = ctCin.ExtractRow(0);  // CA is A^T * s + e 0,A


        // Allocate and copy variables used by functions
	unsigned long long* publicElemB_device;
        cudaMalloc(reinterpret_cast<void **>(&publicElemB_device), sizeof(unsigned long long) * (ell + 1) * m * size * n);
        
	copyMatrixH2D(publicElemB_device,publicElementB);

        unsigned long long* deneme = reinterpret_cast<unsigned long long*>(malloc(sizeof(unsigned long long) * (ell + 1) * m * size * n));
        cudaMemcpy(deneme, publicElemB_device, sizeof(unsigned long long) * (ell + 1) * m * size * n, cudaMemcpyDeviceToHost);

	// THIS LINE CAUSES A PROBLEM
        //printMatrix(publicElemB_device,publicElementB ,ell+1,m,size,n);

        usint* x_device;
        cudaMalloc(&x_device,(ell+1) * sizeof(usint));
        cudaMemcpy(x_device,&x[0], (ell+1) * sizeof(usint),cudaMemcpyHostToDevice);
        unsigned long long* ctCin_device;
        
	// TODO: Bunu tekrar bak
        cudaMalloc(&ctCin_device, sizeof(unsigned long long) * (ell + 1) * m * size * n);
        unsigned long long* evalCf_device;
        cudaMalloc(&ctCin_device, sizeof(unsigned long long) * (1) * m * size * n);

        copyMatrixH2D(ctCin_device, ctCin.ExtractRows(1, ell + 1));


	
        usint* y_device;
        cudaMalloc(&y_device, sizeof(usint));
	//printf("Before entering the EVALCT_GPU kernel \n");
        EvalCT_GPU(sender, ilDCRTParams, publicElementB, publicElemB_device, &x[0], x_device, ctCin_device, ctCin, y_device, evalCf_device, ell, m, size);
	cudaDeviceSynchronize();

        TIC(t1);
        receiver.EvalCT(ilDCRTParams, publicElementB, &x[0],
                        ctCin.ExtractRows(1, ell + 1), &y, &evalCf);
        avg_evalct += TOC(t1);

        TIC(t1);
        pkg.EvalPK(ilDCRTParams, publicElementB, &evalBf);
        avg_evalpk += TOC(t1);

        TIC(t1);
        pkg.KeyGen(ilDCRTParams, trapdoorA.first, evalBf, pubElemBeta,
                   trapdoorA.second, dgg, &sk);
        avg_keygen += TOC(t1);
        //  CheckSecretKeyKPDCRT(m, trapdoorA.first, evalBf, sk, pubElemBeta);

        TIC(t1);
        receiver.Decrypt(ilDCRTParams, sk, ctCA, evalCf, c1, &dtext);
        avg_dec += TOC_US(t1);

        NativeVector ptext2 = ptext.GetValues();
        ptext2.SetModulus(NativeInteger(2));

        if (ptext2 != dtext.GetValues()) {
            std::cout << "Decryption fails at iteration: " << i << std::endl;
            // std::cerr << ptext << std::endl;
            // std::cerr << dtext << std::endl;
            return 0;
        }

        // std::cerr << ptext << std::endl;
        // std::cerr << dtext << std::endl;
    }

    std::cout << "Encryption is successful after " << iter << " iterations!\n";
    std::cout << "Average key generation time : "
              << "\t" << (avg_keygen) / iter << " ms" << std::endl;
    std::cout << "Average ciphertext Format::EVALUATION time : "
              << "\t" << (avg_evalct) / iter << " ms" << std::endl;
    std::cout << "Average public key Format::EVALUATION time : "
              << "\t" << (avg_evalpk) / iter << " ms" << std::endl;
    std::cout << "Average encryption time : "
              << "\t" << (avg_enc) / iter << " ms" << std::endl;
    std::cout << "Average decryption time : "
              << "\t" << (avg_dec) / (iter * 1000) << " ms" << std::endl;

    return 0;
}



usint EvalNANDTree(usint *x, usint ell) {
    usint y;

    if (ell == 2) {
        y = 1 - x[0] * x[1];
        return y;
    } else {
        ell >>= 1;
        y = 1 - (EvalNANDTree(&x[0], ell) * EvalNANDTree(&x[ell], ell));
    }
    return y;
}

