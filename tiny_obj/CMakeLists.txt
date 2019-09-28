set(SOURCE_FILES
	"tiny_obj_loader.h"
	"tiny_obj_loader.cc"
    )

cuda_add_library(tiny_obj
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
