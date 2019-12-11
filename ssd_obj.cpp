/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#include "ssd_obj.hpp"

namespace ssd {
SSDObject::SSDObject(const float& id,
                     const float& confidence,
                     const float& xmin,
                     const float& xmax,
                     const float& ymin,
                     const float& ymax)
    : id(id),
      confidence(confidence),
      xmin(xmin),
      xmax(xmax),
      ymin(ymin),
      ymax(ymax) {}
}  // namespace ssd
