# Go services. Build one with --target gateway or --target evalgate.
FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/evalgate ./cmd/evalgate

FROM gcr.io/distroless/static-debian12:nonroot AS gateway
COPY --from=build /out/gateway /gateway
EXPOSE 8080
ENTRYPOINT ["/gateway"]

# The gate image carries the held-out test set and the released model's metrics, so it
# runs the same way on a laptop, in GitHub Actions and in Jenkins.
FROM gcr.io/distroless/static-debian12:nonroot AS evalgate
WORKDIR /work
COPY --from=build /out/evalgate /evalgate
COPY data/splits/test.csv release-metrics.json ./
ENTRYPOINT ["/evalgate", "-data", "test.csv", "-release", "release-metrics.json", "-url", "http://gateway:8080", "-report", "/tmp/report.json"]
