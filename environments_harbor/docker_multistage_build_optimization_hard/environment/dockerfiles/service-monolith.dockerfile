FROM java:11

WORKDIR /app

COPY . .

RUN ./gradlew dependencies

RUN ./gradlew build

EXPOSE 8080

CMD ["java", "-jar", "build/libs/monolith.jar"]