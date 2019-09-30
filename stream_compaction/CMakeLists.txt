set(SOURCE_FILES
    "common.h"
    "common.cu"
	"shared.h"
	"shared.cu"
"efficient.h"
"efficient.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
