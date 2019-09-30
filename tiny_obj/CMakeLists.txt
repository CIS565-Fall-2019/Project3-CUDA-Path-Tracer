set(SOURCE_FILES
   "tiny_obj_loader.cc"
   "tiny_obj_loader.h"
    )

cuda_add_library(tiny_obj
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
