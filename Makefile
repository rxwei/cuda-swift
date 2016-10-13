OS=$(shell uname)
ifeq ($(OS), Linux)
CFLAGS := -I/usr/lib/cuda/include
LDFLAGS := -L/usr/lib/cuda/lib
else
CFLAGS := -I/usr/local/cuda/include
LDFLAGS := -L/usr/local/cuda/lib
endif

DBGEXE=.build/debug/cool
RELEXE=.build/release/cool

SFLAGS=-Xcc $(CFLAGS) -Xlinker $(LDFLAGS)

all: $(DBGEXE)

$(DBGEXE): Sources/*
	swift build $(SFLAGS)

release: $(RELEXE)

$(RELEXE): Sources/*
	swift build -c release $(SFLAGS)

clean:
	rm -rf $(DBGEXE) $(RELEXE)
	swift build --clean
