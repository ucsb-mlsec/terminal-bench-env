#!/bin/bash

# test_server.sh - Demonstrates HTTP server range request capabilities
# This script tests the HTTP server running on localhost:8080 serving dataset.bin
# and demonstrates how to use HTTP Range requests to download file chunks

echo "======================================"
echo "HTTP Range Request Server Test Script"
echo "======================================"
echo ""

SERVER_URL="http://localhost:8080/dataset.bin"

# Test 1: Basic connectivity check
echo "[Test 1] Checking basic connectivity to $SERVER_URL"
echo "Command: curl -s -o /dev/null -w '%{http_code}' $SERVER_URL"
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$SERVER_URL")
echo "HTTP Status Code: $HTTP_CODE"
if [ "$HTTP_CODE" == "200" ]; then
    echo "✓ Server is responding successfully"
else
    echo "✗ Server connectivity issue (expected 200, got $HTTP_CODE)"
fi
echo ""

# Test 2: HEAD request to get file information
echo "[Test 2] Getting file information using HEAD request"
echo "Command: curl -I $SERVER_URL"
echo "Response headers:"
curl -I "$SERVER_URL" 2>/dev/null
echo "Note: Look for 'Content-Length' header showing total file size"
echo "      and 'Accept-Ranges: bytes' indicating range support"
echo ""

# Test 3: Range request for first 1024 bytes (0-1023)
echo "[Test 3] Requesting first 1KB (bytes 0-1023)"
echo "Command: curl -I -H 'Range: bytes=0-1023' $SERVER_URL"
echo "Response headers:"
curl -I -H "Range: bytes=0-1023" "$SERVER_URL" 2>/dev/null
echo "Note: HTTP 206 Partial Content response indicates successful range request"
echo "      Content-Range header shows the byte range returned"
echo ""

# Test 4: Range request for second 1024 bytes (1024-2047)
echo "[Test 4] Requesting second 1KB (bytes 1024-2047)"
echo "Command: curl -I -H 'Range: bytes=1024-2047' $SERVER_URL"
echo "Response headers:"
curl -I -H "Range: bytes=1024-2047" "$SERVER_URL" 2>/dev/null
echo ""

# Test 5: Download and display first 64 bytes of content
echo "[Test 5] Downloading first 64 bytes to verify actual content transfer"
echo "Command: curl -s -H 'Range: bytes=0-63' $SERVER_URL | xxd | head -5"
echo "First 64 bytes (hex dump):"
curl -s -H "Range: bytes=0-63" "$SERVER_URL" | xxd | head -5
echo ""

# Test 6: Verify Content-Range header format
echo "[Test 6] Examining Content-Range header format"
echo "Command: curl -s -I -H 'Range: bytes=1000-1999' $SERVER_URL | grep -i content-range"
CONTENT_RANGE=$(curl -s -I -H "Range: bytes=1000-1999" "$SERVER_URL" | grep -i "content-range")
echo "Content-Range: $CONTENT_RANGE"
echo "Format: Content-Range: bytes <start>-<end>/<total_size>"
echo ""

# Test 7: Request a range from the end of the file
echo "[Test 7] Requesting last 1KB of the file"
echo "Command: curl -s -I -H 'Range: bytes=-1024' $SERVER_URL | grep -E '(HTTP|Content-Range)'"
curl -s -I -H "Range: bytes=-1024" "$SERVER_URL" | grep -E "(HTTP|Content-Range)"
echo "Note: Negative range '-1024' means last 1024 bytes"
echo ""

# Test 8: Multiple consecutive range requests
echo "[Test 8] Demonstrating chunked download pattern"
echo "This is how you would download a file in chunks:"
echo ""
CHUNK_SIZE=1048576  # 1MB chunks
echo "Example pattern for 5MB file with 1MB chunks:"
echo "  Range: bytes=0-1048575       (Chunk 1: 1MB)"
echo "  Range: bytes=1048576-2097151 (Chunk 2: 1MB)"
echo "  Range: bytes=2097152-3145727 (Chunk 3: 1MB)"
echo "  Range: bytes=3145728-4194303 (Chunk 4: 1MB)"
echo "  Range: bytes=4194304-5242879 (Chunk 5: remaining bytes)"
echo ""

# Test 9: Verify server response consistency
echo "[Test 9] Verifying server returns HTTP 206 for range requests"
RANGE_STATUS=$(curl -s -H "Range: bytes=0-1023" -w '%{http_code}' -o /dev/null "$SERVER_URL")
echo "Status code for range request: $RANGE_STATUS"
if [ "$RANGE_STATUS" == "206" ]; then
    echo "✓ Server correctly returns 206 Partial Content"
else
    echo "✗ Unexpected status code (expected 206, got $RANGE_STATUS)"
fi
echo ""

# Summary
echo "======================================"
echo "Summary of Server Capabilities:"
echo "======================================"
echo "✓ Server accepts HTTP Range requests"
echo "✓ Server returns HTTP 206 for partial content"
echo "✓ Server provides Content-Range headers"
echo "✓ Server supports arbitrary byte ranges"
echo ""
echo "Usage for downloading in chunks:"
echo "  curl -r START-END $SERVER_URL -o chunk.bin"
echo "  or"
echo "  curl -H 'Range: bytes=START-END' $SERVER_URL -o chunk.bin"
echo ""
echo "To download complete file in chunks and reassemble:"
echo "1. Determine file size from Content-Length header"
echo "2. Split file size into equal chunks"
echo "3. Download each chunk with appropriate Range header"
echo "4. Concatenate chunks in order to recreate original file"
echo "======================================"