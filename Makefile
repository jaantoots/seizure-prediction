all:
	cd metrics; luarocks make

clean:
	rm -r metrics/build

remove:
	luarocks remove metrics
