obj-m += tsa_scanning.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

insmod: all
	sudo insmod tsa_scanning.ko

log:
	tail -f /var/log/messages