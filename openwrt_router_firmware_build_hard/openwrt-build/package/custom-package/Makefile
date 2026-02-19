include $(TOPDIR)/rules.mk

PKG_NAME:=custom-router-package
PKG_VERSION:=1.0
PKG_RELEASE:=1

PKG_SOURCE custom-package.tar.gz
PKG_SOURCE_URL:=http://example.com/downloads/
PKG_HASH:=skip

PKG_LICENSE:=GPL-2.0
PKG_LICENSE_FILES:=COPYING

PKG_BUILD_DIR:=$(BUILD_DIR)/$(PKG_NAME)-$(PKG_VERSION)

include $(INCLUDE_DIR)/package.mk

define Package/custom-router-package
  SECTION:=net
  CATEGORY:=Network
  TITLE:=Custom Router Package
  DEPENDS:=+libc +@BROKEN_DEPENDENCY
  URL:=http://example.com/
endef

define Package/custom-router-package/description
  This is a custom router package for OpenWrt firmware.
  It provides additional networking capabilities and configuration options.
endef

define Build/Prepare
	mkdir -p $(PKG_BUILD_DIR)
	$(CP) ./src/* $(PKG_BUILD_DIR)/
endef

define Build/Configure
	$(call Build/Configure/Default)
endef

define Build/Compile
	$(MAKE) -C $(PKG_BUILD_DIR) \
		CC="$(TARGET_CC)" \
		CFLAGS="$(TARGET_CFLAGS)" \
		LDFLAGS="$(TARGET_LDFLAGS)" \
		CUSTOM_VAR="$(UNDEFINED_VAR)" \
		all
    echo "Compiling custom package"
	$(TARGET_CC) -o $(PKG_BUILD_DIR)/custom-router $(PKG_BUILD_DIR)/main.c
endef

define Package/custom-router-package/install
	$(INSTALL_DIR) $(1)/usr/sbin
	$(INSTALL_BIN) $(PKG_BUILD_DIR)/custom-router $(1)/usr/sbin/
	$(INSTALL_DIR) $(1)/etc/config
	$(INSTALL_DATA) ./files/custom-router.config $(1)/etc/config/custom-router
endef

define Package/custom-router-package/preinst
#!/bin/sh
exit 0
endef

define Package/custom-router-package/postinst
#!/bin/sh
/etc/init.d/custom-router enable
exit 0
endef

$(eval $(call BuildPackage,custom-router-package))