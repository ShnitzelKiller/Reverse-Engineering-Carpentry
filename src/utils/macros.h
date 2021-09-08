//
// Created by James Noeckel on 1/23/21.
//

#pragma once

#define LINE_FAIL(msg) if (is_line.fail()) {std::cout << (msg) << "; " << line << std::endl;return false;}
