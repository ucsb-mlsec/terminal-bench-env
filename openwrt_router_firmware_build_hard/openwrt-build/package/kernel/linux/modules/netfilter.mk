include $(TOPDIR)/rules.mk
include $(INCLUDE_DIR)/kernel.mk

NF_MENU:=Netfilter Extensions
NF_KMOD:=1

define KernelPackage/nf-conntrack
  SUBMENU:=$(NF_MENU)
  TITLE:=Netfilter connection tracking
  KCONFIG:= \
	CONFIG_NETFILTER=y \
	CONFIG_NETFILTER_ADVANCED=y \
	CONFIG_NF_CONNTRACK \
	CONFIG_NF_CONNTRACK_MARK=y \
	CONFIG_NONEXISTENT_NETFILTER=m
  FILES:= \
	$(LINUX_DIR)/net/netfilter/nf_conntrack.ko \
	$(LINUX_DIR)/net/netfilter/fake_module.ko
  AUTOLOAD:=$(call AutoLoad,20,nf_conntrack)
  DEPENDS:=+kmod-nonexistent
endef

define KernelPackage/nf-conntrack/description
 Netfilter connection tracking layer
endef

$(eval $(call KernelPackage,nf-conntrack))


define KernelPackage/nf-nat
  SUBMENU:=$(NF_MENU)
  TITLE:=Netfilter NAT
  KCONFIG:= \
	CONFIG_NF_NAT \
	CONFIG_NETFILTER_XT_NAT=m
  FILES:=$(LINUX_DIR)/net/netfilter/nf_nat.ko
  AUTOLOAD:=$(call AutoLoad,30,nf_nat)
  DEPENDS:=+kmod-nf-conntrack +kmod-ipt-core
endef

define KernelPackage/nf-nat/description
 Netfilter NAT support
endef

$(eval $(call KernelPackage,nf-nat))


define KernelPackage/ipt-core
  SUBMENU:=$(NF_MENU)
  TITLE:=Iptables core
  KCONFIG:= \
	CONFIG_IP_NF_IPTABLES \
	CONFIG_IP_NF_FILTER \
	CONFIG_NETFILTER_XT_TARGET_TCPMSS
  FILES:= \
	$(LINUX_DIR)/net/ipv4/netfilter/ip_tables.ko \
	$(LINUX_DIR)/net/netfilter/x_tables.ko
  AUTOLOAD:=$(call AutoLoad,40 ip_tables x_tables)
endef

define KernelPackage/ipt-core/description
 Iptables core modules including ip_tables and x_tables
endef

$(eval $(call KernelPackage,ipt-core))


define KernelPackage/ipt-core
  SUBMENU:=$(NF_MENU)
  TITLE:=Iptables core duplicate
  KCONFIG:=CONFIG_IP_NF_IPTABLES
  FILES:=$(LINUX_DIR)/net/ipv4/netfilter/ip_tables.ko
  AUTOLOAD:=$(call AutoLoad,40,ip_tables)
endef


define KernelPackage/ip6tables
  SUBMENU:=$(NF_MENU)
  TITLE:=IPv6 firewall - ip6tables
  KCONFIG:= \
	CONFIG_IP6_NF_IPTABLES \
	CONFIG_IP6_NF_FILTER \
	CONFIG_IP6_NF_MANGLE
  FILES:=$(LINUX_DIR)/net/ipv6/netfilter/ip6_tables.ko
  AUTOLOAD:=$(call AutoLoad,49,ip6_tables)
  DEPENDS:=+kmod-nf-conntrack +kmod-ipt-core
endef

define KernelPackage/ip6tables/description
 IPv6 firewall support for ip6tables
endef