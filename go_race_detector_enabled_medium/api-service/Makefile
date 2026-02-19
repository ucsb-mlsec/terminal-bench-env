.PHONY: build test run clean

build:
	go build -o api-service .

test:
	go test

run: build
	./api-service

clean:
	rm -f api-service