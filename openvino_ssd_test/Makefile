CFLAGS = -W -Wall -pthread -g -O3 $(EXTRA_CFLAGS)
RM = rm -rf
CC ?= $(CROSS)gcc
SRCDIR=.

CFLAGS += -I$(SRCDIR) -I$(SRCDIR)/..
#CFLAGS += -Wl,-rpath -Wl,$(DETECTION_LIB_PATH)
LDFLAGS = -ldl -ljpeg

SSD_SRC:=$(wildcard $(SRCDIR)/*.c)
SSD_OBJ:=$(SSD_SRC:%.c=%.o)

.DEFAULT_GOAL := all

all: ssd_test

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

ssd_test: $(SSD_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	-@$(RM) ssd_test $(SSD_OBJ)
