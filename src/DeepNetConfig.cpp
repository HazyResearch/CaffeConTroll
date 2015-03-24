
#include "DeepNetConfig.h"

// Declare train_ here, since it's a static variable (linker error otherwise)
// We set it to true, since we want to train by default. This is also useful
// for testing (e.g. DropoutTest).
bool DeepNetConfig::train_ = true;

