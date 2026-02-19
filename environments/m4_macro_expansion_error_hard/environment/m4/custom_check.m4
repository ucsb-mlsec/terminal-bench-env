dnl Custom feature detection macro
dnl This macro checks for custom feature support

AC_DEFUN([CUSTOM_CHECK_FEATURE], [
  AC_MSG_CHECKING([for custom feature support])
  has_feature=yes
  AC_MSG_RESULT([$has_feature])
  AC_DEFINE([HAVE_CUSTOM_FEATURE], [1], [Define if custom feature is available])
  dnl Missing closing bracket - should be ] before )
)

dnl End of custom_check.m4