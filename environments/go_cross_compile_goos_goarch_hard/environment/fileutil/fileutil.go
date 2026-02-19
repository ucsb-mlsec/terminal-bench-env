package fileutil

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"time"
)

// FileMetadata contains information about a file
type FileMetadata struct {
	Path         string
	Size         int64
	Mode         os.FileMode
	ModTime      time.Time
	Inode        uint64
	IsRegular    bool
	LinkCount    uint64
}

// ProcessFiles walks through a directory and processes all files
func ProcessFiles(rootDir string) ([]FileMetadata, error) {
	var results []FileMetadata
	
	err := filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		metadata, err := GetFileMetadata(path)
		if err != nil {
			fmt.Printf("Warning: could not get metadata for %s: %v\n", path, err)
			return nil
		}
		
		results = append(results, metadata)
		return nil
	})
	
	return results, err
}

// GetFileInfo returns basic file information that works cross-platform
func GetFileInfo(path string) (os.FileInfo, error) {
	return os.Stat(path)
}

// GetFileMetadata returns detailed metadata about a file
func GetFileMetadata(path string) (FileMetadata, error) {
	info, err := os.Stat(path)
	if err != nil {
		return FileMetadata{}, err
	}
	
	metadata := FileMetadata{
		Path:    path,
		Size:    info.Size(),
		Mode:    info.Mode(),
		ModTime: info.ModTime(),
	}
	
	// Get inode information - this uses platform-specific code
	inode, err := GetFileInode(path)
	if err == nil {
		metadata.Inode = inode
	}
	
	// Check if it's a regular file using platform-specific syscall
	metadata.IsRegular = IsRegularFile(path)
	
	// Get link count
	linkCount, err := GetLinkCount(path)
	if err == nil {
		metadata.LinkCount = linkCount
	}
	
	return metadata, nil
}

// GetFileInode returns the inode number of a file
// This directly accesses Unix-specific fields without build constraints
func GetFileInode(path string) (uint64, error) {
	var stat syscall.Stat_t
	err := syscall.Stat(path, &stat)
	if err != nil {
		return 0, err
	}
	
	// Direct access to Ino field - doesn't exist on Windows
	return stat.Ino, nil
}

// GetLinkCount returns the number of hard links to a file
func GetLinkCount(path string) (uint64, error) {
	var stat syscall.Stat_t
	err := syscall.Stat(path, &stat)
	if err != nil {
		return 0, err
	}
	
	// Nlink field is Unix-specific
	return uint64(stat.Nlink), nil
}

// IsRegularFile checks if a file is a regular file using syscall
func IsRegularFile(path string) bool {
	var stat syscall.Stat_t
	err := syscall.Stat(path, &stat)
	if err != nil {
		return false
	}
	
	// S_IFREG is a Unix-specific constant
	return (stat.Mode & syscall.S_IFREG) != 0
}

// GetDirectoryEntries uses low-level syscall to read directory entries
// This uses Linux-specific syscall numbers
func GetDirectoryEntries(dirPath string) ([]string, error) {
	fd, err := syscall.Open(dirPath, syscall.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	defer syscall.Close(fd)
	
	var entries []string
	buf := make([]byte, 4096)
	
	// Using Linux-specific syscall number for getdents
	for {
		n, err := syscall.Syscall(syscall.SYS_GETDENTS64, uintptr(fd), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
		if err != 0 {
			return nil, err
		}
		if n == 0 {
			break
		}
		
		// Parse directory entries from buffer
		// This is simplified and wouldn't actually work properly
		entries = append(entries, "entry")
	}
	
	return entries, nil
}

// ChangeFileOwner changes the owner of a file
// Uses Unix-specific syscall
func ChangeFileOwner(path string, uid, gid int) error {
	return syscall.Chown(path, uid, gid)
}

// GetFileUID returns the user ID of the file owner
func GetFileUID(path string) (uint32, error) {
	var stat syscall.Stat_t
	err := syscall.Stat(path, &stat)
	if err != nil {
		return 0, err
	}
	
	// Uid field is Unix-specific
	return stat.Uid, nil
}

// GetFileGID returns the group ID of the file
func GetFileGID(path string) (uint32, error) {
	var stat syscall.Stat_t
	err := syscall.Stat(path, &stat)
	if err != nil {
		return 0, err
	}
	
	// Gid field is Unix-specific
	return stat.Gid, nil
}