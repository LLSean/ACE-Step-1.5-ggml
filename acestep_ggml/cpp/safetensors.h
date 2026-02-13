#ifndef ACESTEP_SAFETENSORS_H
#define ACESTEP_SAFETENSORS_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ace_safetensors {

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_start = 0;
    uint64_t data_end = 0;

    size_t nbytes() const;
};

struct File {
    std::string path;
    uint64_t header_size = 0;
    uint64_t data_offset = 0;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, size_t> index;

    bool load(const std::string & file_path, std::string & error);
    const TensorInfo * find(const std::string & name) const;
    bool read_tensor(const TensorInfo & info, void * dst, size_t dst_size, std::string & error) const;
};

size_t dtype_size_bytes(const std::string & dtype, bool & ok);

}  // namespace ace_safetensors

#endif  // ACESTEP_SAFETENSORS_H
