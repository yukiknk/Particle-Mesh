# コンパイラを指定 (MPI用)
CXX = mpiFCCpx
FC = mpifrtpx

# コンパイルオプションを指定
CXXFLAGS = -Kfast,openmp -std=c++17 -I./include
FFLAGS = -Kfast,openmp

# FFTWおよびFFTE(MPI)ライブラリへのリンクオプション
LIBS = -lfftw3_mpi -lfftw3_omp -lfftw3_threads -lfftw3 -lm -lmpi_mpifh -lmpi

# ターゲットの実行ファイル名
TARGET = pm_fft_test

# ソースファイル
SRCS = src/main.cpp \
       src/interpolater.cpp \
       src/grouping.cpp \
       src/fft/fft_fftw.cpp \
       src/fft/fft_ffte1.cpp \
       src/fft/fft_ffte2.cpp \
       src/transpose/transpose_fwd_slab.cpp \
       src/transpose/transpose_bwd_slab.cpp \
       src/transpose/transpose_fwd_pencil.cpp \
       src/transpose/transpose_bwd_pencil.cpp

FSRCS = src/fft/factor.f \
        src/fft/fft235.f \
        src/fft/kernel.f \
        src/fft/pdzfft3d.f \
        src/fft/pzdfft3d.f \
        src/fft/pdzfft3dv.f \
        src/fft/pzdfft3dv.f

# オブジェクトファイル（src/以下の構造を維持）
CPP_OBJS = $(SRCS:.cpp=.o)
F_OBJS = $(FSRCS:.f=.o)
OBJS = $(CPP_OBJS) $(F_OBJS)

# デフォルトターゲット
all: $(TARGET)

# 実行ファイルのビルドルール
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

# ソースファイルからオブジェクトファイルを作成
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.f
	$(FC) $(FFLAGS) -c $< -o $@

# クリーンアップ
clean:
	rm -f $(OBJS) $(TARGET)

# デバッグビルド（デバッグ出力を有効化）
debug: CXXFLAGS += -DDEBUG_MODE
debug: $(TARGET)

# コピー（必要に応じてパスを変更）
copy:
	cp ./$(TARGET) /vol0005/mdt0/data/hp230173/u13308/All/

.PHONY: all clean copy debug