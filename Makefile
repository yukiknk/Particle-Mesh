# コンパイラを指定 (MPI用)
CXX = mpiFCCpx

# コンパイルオプションを指定
CXXFLAGS = -Kfast,openmp -std=c++17 -I./include

# FFTWライブラリへのリンクオプション
LIBS = -lfftw3_mpi -lfftw3_omp -lfftw3_threads -lfftw3 -lm

# ターゲットの実行ファイル名
TARGET = pm_fftw

# ソースファイル
SRCS = src/main.cpp \
       src/interpolater.cpp \
       src/fft/fft_fftw.cpp \
       src/transpose/transpose_slab_fwd.cpp

# オブジェクトファイル（src/以下の構造を維持）
OBJS = $(SRCS:.cpp=.o)

# デフォルトターゲット
all: $(TARGET)

# 実行ファイルのビルドルール
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

# ソースファイルからオブジェクトファイルを作成
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

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
