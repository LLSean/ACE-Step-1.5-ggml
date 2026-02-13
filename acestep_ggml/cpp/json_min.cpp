#include "json_min.h"

#include <cctype>

namespace ace_json {

void Parser::skip_ws() {
    while (cur_ < end_ && std::isspace(static_cast<unsigned char>(*cur_))) {
        ++cur_;
    }
}

bool Parser::match(const char * s) {
    const char * p = s;
    const char * c = cur_;
    while (*p) {
        if (c >= end_ || *c != *p) {
            return false;
        }
        ++p;
        ++c;
    }
    cur_ = c;
    return true;
}

Value Parser::parse() {
    skip_ws();
    Value v = parse_value();
    skip_ws();
    if (cur_ != end_) {
        throw std::runtime_error("json: trailing characters");
    }
    return v;
}

Value Parser::parse_value() {
    skip_ws();
    if (cur_ >= end_) {
        throw std::runtime_error("json: unexpected end of input");
    }
    char c = *cur_;
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string();
    if (c == 't') {
        if (!match("true")) throw std::runtime_error("json: invalid token");
        Value v; v.type = Type::Bool; v.boolean = true; return v;
    }
    if (c == 'f') {
        if (!match("false")) throw std::runtime_error("json: invalid token");
        Value v; v.type = Type::Bool; v.boolean = false; return v;
    }
    if (c == 'n') {
        if (!match("null")) throw std::runtime_error("json: invalid token");
        Value v; v.type = Type::Null; return v;
    }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
    throw std::runtime_error("json: invalid value");
}

Value Parser::parse_string() {
    Value v;
    v.type = Type::String;
    if (*cur_ != '"') {
        throw std::runtime_error("json: expected string");
    }
    ++cur_;
    std::string out;
    while (cur_ < end_) {
        char c = *cur_++;
        if (c == '"') {
            v.string = out;
            return v;
        }
        if (c == '\\') {
            if (cur_ >= end_) throw std::runtime_error("json: invalid escape");
            char e = *cur_++;
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u':
                    // minimal Unicode escape handling: skip 4 hex digits
                    if (end_ - cur_ < 4) throw std::runtime_error("json: invalid unicode escape");
                    cur_ += 4;
                    out.push_back('?');
                    break;
                default:
                    throw std::runtime_error("json: invalid escape");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("json: unterminated string");
}

Value Parser::parse_number() {
    const char * start = cur_;
    if (*cur_ == '-') ++cur_;
    if (cur_ >= end_ || (!std::isdigit(static_cast<unsigned char>(*cur_)) && *cur_ != '.')) {
        throw std::runtime_error("json: invalid number");
    }
    while (cur_ < end_ && std::isdigit(static_cast<unsigned char>(*cur_))) ++cur_;
    if (cur_ < end_ && *cur_ == '.') {
        ++cur_;
        while (cur_ < end_ && std::isdigit(static_cast<unsigned char>(*cur_))) ++cur_;
    }
    if (cur_ < end_ && (*cur_ == 'e' || *cur_ == 'E')) {
        ++cur_;
        if (cur_ < end_ && (*cur_ == '+' || *cur_ == '-')) ++cur_;
        if (cur_ >= end_ || !std::isdigit(static_cast<unsigned char>(*cur_))) {
            throw std::runtime_error("json: invalid exponent");
        }
        while (cur_ < end_ && std::isdigit(static_cast<unsigned char>(*cur_))) ++cur_;
    }

    std::string s(start, cur_);
    Value v;
    v.type = Type::Number;
    v.number = std::stod(s);
    return v;
}

Value Parser::parse_array() {
    Value v;
    v.type = Type::Array;
    if (*cur_ != '[') throw std::runtime_error("json: expected array");
    ++cur_;
    skip_ws();
    if (cur_ < end_ && *cur_ == ']') {
        ++cur_;
        return v;
    }
    while (cur_ < end_) {
        v.array.push_back(parse_value());
        skip_ws();
        if (cur_ >= end_) break;
        if (*cur_ == ',') {
            ++cur_;
            continue;
        }
        if (*cur_ == ']') {
            ++cur_;
            return v;
        }
        throw std::runtime_error("json: expected ',' or ']'");
    }
    throw std::runtime_error("json: unterminated array");
}

Value Parser::parse_object() {
    Value v;
    v.type = Type::Object;
    if (*cur_ != '{') throw std::runtime_error("json: expected object");
    ++cur_;
    skip_ws();
    if (cur_ < end_ && *cur_ == '}') {
        ++cur_;
        return v;
    }
    while (cur_ < end_) {
        skip_ws();
        Value key = parse_string();
        skip_ws();
        if (cur_ >= end_ || *cur_ != ':') throw std::runtime_error("json: expected ':'");
        ++cur_;
        Value val = parse_value();
        v.object.emplace(key.string, std::move(val));
        skip_ws();
        if (cur_ >= end_) break;
        if (*cur_ == ',') {
            ++cur_;
            continue;
        }
        if (*cur_ == '}') {
            ++cur_;
            return v;
        }
        throw std::runtime_error("json: expected ',' or '}'");
    }
    throw std::runtime_error("json: unterminated object");
}

}  // namespace ace_json
