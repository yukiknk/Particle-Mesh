#include <array>

int main(int argc, char** argv) {
    //MPIクラス初期化

    //バッファ初期化
    //Grid初期化
    //FFT初期化
    //Transpose_FWD初期化
    //Transpose_BWD初期化
    //Particle初期化
    //Interpolater初期化
    
    //バッファ確保

    //ループのセットアップ
    std::array<double, 9> exe_time;
    exe_time.fill(0.0);
    const int warm_up = 2;
    const int loop = 10;
    const int all_loop = warm_up + loop;
    double a = 0.9;

    //メインループ
    for (int i = 0; i < all_loop; i++) {
        //cloud in cell

        //pack
        //alltoallv
        //unpack

        //FFT
        //Green
        //IFFT

        //pack
        //alltoallv
        //unpack

        //update particle
    }

    //時間出力
}
