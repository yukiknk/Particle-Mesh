DECOMP_CHOICES := slab pencil cube       # 分割アルゴリズムの選択肢
FFT_CHOICES    := fftw ffte1 ffte2           # FFT ライブラリの選択肢

# ユーザがターゲットに指定した単語を取得
DECOMP_GOAL := $(firstword $(filter $(DECOMP_CHOICES),$(MAKECMDGOALS)))
FFT_GOAL    := $(firstword $(filter $(FFT_CHOICES),$(MAKECMDGOALS)))

# デフォルト値（CLI で上書き可）
DECOMP ?= $(if $(DECOMP_GOAL),$(DECOMP_GOAL),slab)
FFT    ?= $(if $(FFT_GOAL),$(FFT_GOAL),fftw)

# 疑似ターゲットを除外して make がファイルを探さないようにする
MAKECMDGOALS := $(filter-out $(DECOMP_CHOICES) $(FFT_CHOICES),$(MAKECMDGOALS))

# これらの名前を phony ターゲットとして登録（処理なし）
.PHONY: $(DECOMP_CHOICES) $(FFT_CHOICES)

# デフォルトゴールを設定
.DEFAULT_GOAL := all

CXX = mpiFCCpx
FC  = mpifrtpx

CXXFLAGS_BASE  := -Kfast,openmp -std=c++17
CXXFLAGS       := $(CXXFLAGS_BASE)

FFLAGS_BASE   := -Kfast,openmp
FFLAGS        := $(FFLAGS_BASE)
LDFLAGS       := -Kopenmp

ifeq ($(FFT),fftw)
  CXXFLAGS := $(CXXFLAGS_BASE) -DUSE_FFTW
  FFT_LIBS := -lfftw3_mpi -lfftw3_omp -lfftw3
  LD := $(CXX)
else ifeq ($(FFT),ffte1)
  CXXFLAGS := $(CXXFLAGS_BASE)
  FFT_LIBS :=
  LD := $(CXX)
  LDFLAGS += --linkfortran
  FPPFLAGS ?= -Cpp
else
  $(error 対応していない FFT ライブラリ: $(FFT))
endif

INCLUDES := -Iinclude -Iinclude/utils -Iinclude/decomp/$(DECOMP) -Iinclude/fft/$(FFT) -Iinclude/decomp -Iinclude/fft

LIBS := $(FFT_LIBS) -lm -lrt

CORE_SRCS    := $(wildcard src/core/*.cpp)
UTIL_SRCS    := $(wildcard src/utils/*.cpp)
DECOMP_SRCS  := $(wildcard src/decomp/$(DECOMP)/*.cpp)
FFT_CPP_SRCS := $(wildcard src/fft/$(FFT)/*.cpp)

ifeq ($(FFT),ffte1)
  FFT_F_SRCS := $(wildcard src/fft/$(FFT)/*.f)
else
  FFT_F_SRCS :=
endif

CPP_SRCS := $(CORE_SRCS) $(UTIL_SRCS) $(DECOMP_SRCS) $(FFT_CPP_SRCS)

OBJDIR   := build
CPPOBJS  := $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))
FOBJS    := $(patsubst src/%.f,$(OBJDIR)/%.o,$(FFT_F_SRCS))
OBJS     := $(CPPOBJS) $(FOBJS)

TARGET := $(FFT)

.PHONY: all clean copy

all: $(TARGET)

CPPFLAGS := $(INCLUDES)

# C++
$(OBJDIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Fortran (.f 固定形式)
$(OBJDIR)/%.o: src/%.f
	@mkdir -p $(dir $@)
	$(FC) $(FFLAGS) $(CPPFLAGS) $(FPPFLAGS) -c $< -o $@

# リンク（Fortran混在時はLD=mpifrtpx）
$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

clean:
	rm -rf $(OBJDIR) $(TARGET)

DEST := /vol0005/mdt0/data/hp230173/u13308/All
copy: $(TARGET)
	rm -f $(DEST)/$(TARGET)
	cp $(TARGET) $(DEST)
	
