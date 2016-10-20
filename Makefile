OS=$(shell uname)
ifeq ($(OS), Linux)
INCDIR := /usr/local/cuda/include
LIBDIR := /usr/local/cuda/lib
else
INCDIR := /usr/local/cuda/include
LIBDIR := /usr/local/cuda/lib
endif

CFLAGS=-I$(INCDIR)
LDFLAGS=-L$(LIBDIR)
SFLAGS=-Xcc $(CFLAGS) -Xlinker $(LDFLAGS)

all:
	swift build $(SFLAGS)

release:
	swift build -c release $(SFLAGS)

test:
	swift test $(SFLAGS)

update:
	swift package update

xcodeproj:
	swift package generate-xcodeproj
	echo 'HEADER_SEARCH_PATHS = $(INCDIR)'\
     '\nLIBRARY_SEARCH_PATHS = $(LIBDIR)'\
     '\nLD_RUNPATH_SEARCH_PATHS = $(LIBDIR)'\
    >> $(wildcard *.xcodeproj)/Configs/Project.xcconfig

clean:
	swift build --clean
