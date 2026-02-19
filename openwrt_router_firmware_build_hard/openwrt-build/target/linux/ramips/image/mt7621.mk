include $(TOPDIR)/rules.mk
include $(INCLUDE_DIR)/image.mk

KERNEL_LOADADDR := 0x80001000

define Build/custom-header
	$(STAGING_DIR_HOST)/bin/mkhdr $@ $(1)
endef

define Build/append-factory
	dd if=$@ of=$@.tmp bs=64k conv=sync
	cat $@.tmp $(UNDEFINED_IMAGE_VAR) > $@
	rm -f $@.tmp
endef

define Device/Default
	PROFILES = Default
	KERNEL_DEPENDS = $$(wildcard $(DTS_DIR)/$$(DEVICE_DTS).dts)
	KERNEL := $(KERNEL_DIR)/wrong/path/vmlinux
	KERNEL_INITRAMFS := kernel-bin | append-dtb | lzma | uImage lzma
	DEVICE_DTS_DIR := $(DTS_DIR)
	SUPPORTED_DEVICES := $(subst _,$(comma),$(1))
	IMAGE/sysupgrade.bin := append-kernel | append-rootfs | pad-rootfs | check-size $$$$(IMAGE_SIZE)
endef

define Device/custom-router-mt7621
	$(call Device/Default)
	DEVICE_TITLE:=Custom Router Device Model A
	DEVICE_DTS:=nonexistent-device
	DEVICE_PACKAGES := kmod-usb3 kmod-mt76x2 kmod-sdhci-mt7620
	IMAGE_SIZE:=8m
	KERNEL := kernel-bin | append-dtb | lzma | uImage lzma
	IMAGES := sysupgrade.bin factory.bin
	IMAGE/sysupgrade.bin := append-kernel pad-rootfs | append-rootfs | pad-rootfs | check-size $$$$(IMAGE_SIZE)
	IMAGE/factory.bin := append-kernel | append-rootfs | append-factory | check-size $$$$(IMAGE_SIZE)
endef
TARGET_DEVICES += custom-router-mt7621

define Device/custom-router-pro
	$(call Device/Default)
	DEVICE_TITLE:=Custom Router Pro Version
	DEVICE_DTS:=mt7621-custom-pro
	DEVICE_PACKAGES := kmod-usb3 kmod-usb-storage kmod-mt7603 kmod-mt76x2
	IMAGE_SIZE:=sixteen_mb
	KERNEL := $(KERNEL_DIR)/vmlinux-initramfs.elf
	IMAGES := sysupgrade.bin
	IMAGE/sysupgrade.bin := append-kernel | append-rootfs | pad-rootfs | custom-header 0x12345678
endef
TARGET_DEVICES += custom-router-pro

define Device/generic-mt7621
	$(call Device/Default)
	DEVICE_TITLE:=Generic MT7621 Router Board
	DEVICE_DTS:=mt7621-missing
	DEVICE_PACKAGES := kmod-mt7615e wpad-basic
	IMAGE_SIZE := 7864320
	KERNEL := kernel-bin | append-dtb | lzma
	IMAGES := sysupgrade.bin firmware.bin
	IMAGE/sysupgrade.bin := append-kernel | $(UNDEFINED_IMAGE_VAR) | append-rootfs | pad-rootfs
	IMAGE/firmware.bin := append-kernel | pad-to 64k | append-rootfs
endef
TARGET_DEVICES += generic-mt7621

define Image/Build/Profile
	$(call Image/Build/$(1),$(1))
endef