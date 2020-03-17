#ifndef _CPU_UTILS_H_
#define _CPU_UTILS_H_

#include <stdint.h>

typedef struct kernel_info {
    char *name;
    size_t param_size;
    size_t param_num;
    uint16_t *param_offsets;
    uint16_t *param_sizes;
    void *host_fun;
} kernel_info_t;

void kernel_infos_free(kernel_info_t *infos, size_t kernelnum);


void* cricketd_utils_symbol_address(char *symbol);
int cricketd_utils_launch_child(const char *file, char **args);
int cricketd_utils_parameter_size(kernel_info_t **infos, size_t *kernelnum);
kernel_info_t* cricketd_utils_search_info(kernel_info_t *infos, size_t kernelnum, char *kernelname);

#endif //_CPU_UTILS_H_