#include "safetensors.h"

#include "json_min.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ace_safetensors {

static bool read_u64_le(std::ifstream & in, uint64_t & out) {
    uint8_t buf[8];
    if (!in.read(reinterpret_cast<char *>(buf), sizeof(buf))) {
        return false;
    }
    out = 0;
    for (int i = 7; i >= 0; --i) {
        out = (out << 8) | buf[i];
    }
    return true;
}

size_t dtype_size_bytes(const std::string & dtype, bool & ok) {
    ok = true;
    if (dtype == "F16" || dtype == "BF16") return 2;
    if (dtype == "F32") return 4;
    if (dtype == "I32") return 4;
    if (dtype == "I64") return 8;
    ok = false;
    return 0;
}

size_t TensorInfo::nbytes() const {
    bool ok = false;
    size_t elem = dtype_size_bytes(dtype, ok);
    if (!ok) return 0;
    uint64_t count = 1;
    for (auto d : shape) {
        count *= static_cast<uint64_t>(d);
    }
    return static_cast<size_t>(count * elem);
}

bool File::load(const std::string & file_path, std::string & error) {
    path = file_path;
    tensors.clear();
    index.clear();
    header_size = 0;
    data_offset = 0;

    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        error = "failed to open file";
        return false;
    }

    if (!read_u64_le(in, header_size)) {
        error = "failed to read header size";
        return false;
    }

    std::string header_json;
    header_json.resize(static_cast<size_t>(header_size));
    if (!in.read(header_json.data(), static_cast<std::streamsize>(header_size))) {
        error = "failed to read header";
        return false;
    }

    data_offset = 8 + header_size;

    try {
        ace_json::Parser parser(header_json);
        ace_json::Value root = parser.parse();
        if (!root.is_object()) {
            error = "invalid safetensors header";
            return false;
        }

        for (const auto & kv : root.as_object()) {
            const std::string & name = kv.first;
            if (name == "__metadata__") {
                continue;
            }
            const ace_json::Value & obj = kv.second;
            if (!obj.is_object()) {
                error = "tensor entry is not object";
                return false;
            }

            const auto & fields = obj.as_object();
            auto it_dtype = fields.find("dtype");
            auto it_shape = fields.find("shape");
            auto it_offsets = fields.find("data_offsets");
            if (it_dtype == fields.end() || it_shape == fields.end() || it_offsets == fields.end()) {
                error = "missing tensor fields";
                return false;
            }

            TensorInfo info;
            info.name = name;
            info.dtype = it_dtype->second.as_string();

            if (!it_shape->second.is_array()) {
                error = "shape is not array";
                return false;
            }
            for (const auto & v : it_shape->second.as_array()) {
                info.shape.push_back(v.as_int());
            }

            if (!it_offsets->second.is_array() || it_offsets->second.as_array().size() != 2) {
                error = "data_offsets invalid";
                return false;
            }
            info.data_start = static_cast<uint64_t>(it_offsets->second.as_array()[0].as_int());
            info.data_end = static_cast<uint64_t>(it_offsets->second.as_array()[1].as_int());

            index.emplace(info.name, tensors.size());
            tensors.push_back(std::move(info));
        }
    } catch (const std::exception & ex) {
        error = ex.what();
        return false;
    }

    return true;
}

const TensorInfo * File::find(const std::string & name) const {
    auto it = index.find(name);
    if (it == index.end()) {
        return nullptr;
    }
    return &tensors[it->second];
}

bool File::read_tensor(const TensorInfo & info, void * dst, size_t dst_size, std::string & error) const {
    if (!dst || dst_size == 0) {
        error = "invalid destination";
        return false;
    }
    size_t expected = info.nbytes();
    if (expected == 0 || expected != dst_size) {
        error = "size mismatch";
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "failed to open file";
        return false;
    }

    uint64_t offset = data_offset + info.data_start;
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        error = "seek failed";
        return false;
    }

    if (!in.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(dst_size))) {
        error = "read failed";
        return false;
    }

    return true;
}

}  // namespace ace_safetensors
