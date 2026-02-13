#include "qwen_config.h"

#include "json_min.h"

#include <fstream>
#include <sstream>

namespace ace_qwen {

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream in(path);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

bool load_config(const std::string & path, Config & cfg, std::string & error) {
    std::string data;
    if (!read_file(path, data)) {
        error = "failed to read config";
        return false;
    }
    try {
        ace_json::Parser parser(data);
        ace_json::Value root = parser.parse();
        if (!root.is_object()) {
            error = "config is not object";
            return false;
        }
        const auto & obj = root.as_object();
        auto get_int = [&](const char * key, int32_t & dst) -> bool {
            auto it = obj.find(key);
            if (it == obj.end()) return false;
            dst = static_cast<int32_t>(it->second.as_int());
            return true;
        };
        auto get_float = [&](const char * key, float & dst) -> bool {
            auto it = obj.find(key);
            if (it == obj.end()) return false;
            dst = static_cast<float>(it->second.as_number());
            return true;
        };
        auto get_str = [&](const char * key, std::string & dst) -> bool {
            auto it = obj.find(key);
            if (it == obj.end()) return false;
            dst = it->second.as_string();
            return true;
        };

        if (!get_int("vocab_size", cfg.vocab_size)) return false;
        if (!get_int("hidden_size", cfg.hidden_size)) return false;
        if (!get_int("num_hidden_layers", cfg.num_hidden_layers)) return false;
        if (!get_int("num_attention_heads", cfg.num_attention_heads)) return false;
        if (!get_int("num_key_value_heads", cfg.num_key_value_heads)) return false;
        if (!get_int("intermediate_size", cfg.intermediate_size)) return false;
        if (!get_int("head_dim", cfg.head_dim)) return false;
        if (!get_int("max_position_embeddings", cfg.max_position_embeddings)) return false;
        if (!get_float("rms_norm_eps", cfg.rms_norm_eps)) return false;
        (void)get_float("rope_theta", cfg.rope_theta);
        (void)get_str("dtype", cfg.dtype);
    } catch (const std::exception & ex) {
        error = ex.what();
        return false;
    }
    return true;
}

}  // namespace ace_qwen
