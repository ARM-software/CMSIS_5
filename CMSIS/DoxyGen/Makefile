all: clean doc copy_search_css

doc:
	make -C Core
	make -C DAP
	make -C DSP
	make -C Driver
	make -C General
	make -C Pack
	make -C RTOS
	make -C RTOS2
	make -C SVD

clean:
	rm -rf ../Documentation/Core
	rm -rf ../Documentation/DAP
	rm -rf ../Documentation/DSP
	rm -rf ../Documentation/Driver
	rm -rf ../Documentation/General
	rm -rf ../Documentation/Pack
	rm -rf ../Documentation/RTOS
	rm -rf ../Documentation/RTOS2
	rm -rf ../Documentation/SVD

copy_search_css:
	cp Doxygen_Templates/search.css ../Documentation/Core/html/search/.
	cp Doxygen_Templates/search.css ../Documentation/DAP/html/search/.
	cp Doxygen_Templates/search.css ../Documentation/DSP/html/search/.
	cp Doxygen_Templates/search.css ../Documentation/Driver/html/search/.
	#cp Doxygen_Templates/search.css ../Documentation/General/html/search/.
	cp Doxygen_Templates/search.css ../Documentation/Pack/html/search/.
	cp Doxygen_Templates/search.css ../Documentation/RTOS/html/search/.
	cp Doxygen_Templates/search.css ../Documentation/RTOS2/html/search/.
	#cp Doxygen_Templates/search.css ../Documentation/SVD/html/search/.

.PHONY: all clean copy_search_css doc
