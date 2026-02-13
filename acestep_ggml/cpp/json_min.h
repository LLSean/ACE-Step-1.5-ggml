#ifndef ACESTEP_JSON_MIN_H
#define ACESTEP_JSON_MIN_H

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ace_json {

enum class Type {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
};

struct Value {
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    const std::string & as_string() const {
        if (!is_string()) throw std::runtime_error("json: not a string");
        return string;
    }
    int64_t as_int() const {
        if (!is_number()) throw std::runtime_error("json: not a number");
        return static_cast<int64_t>(number);
    }
    double as_number() const {
        if (!is_number()) throw std::runtime_error("json: not a number");
        return number;
    }
    const std::vector<Value> & as_array() const {
        if (!is_array()) throw std::runtime_error("json: not an array");
        return array;
    }
    const std::map<std::string, Value> & as_object() const {
        if (!is_object()) throw std::runtime_error("json: not an object");
        return object;
    }
};

class Parser {
public:
    explicit Parser(const std::string & input) : data_(input), cur_(data_.c_str()), end_(data_.c_str() + data_.size()) {}

    Value parse();

private:
    const std::string data_;
    const char * cur_;
    const char * end_;

    void skip_ws();
    Value parse_value();
    Value parse_object();
    Value parse_array();
    Value parse_string();
    Value parse_number();
    bool match(const char * s);
};

}  // namespace ace_json

#endif  // ACESTEP_JSON_MIN_H
