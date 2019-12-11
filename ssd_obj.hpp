/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#pragma once

namespace ssd {
struct SSDObject {
    SSDObject(const float& id = -1,
              const float& confidence = 0,
              const float& xmin = 0,
              const float& xmax = 0,
              const float& ymin = 0,
              const float& ymax = 0);

    float id;
    float confidence;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
};
}  // namespace ssd
