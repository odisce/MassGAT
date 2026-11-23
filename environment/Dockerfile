ARG PIXI_VERSION=0.75.0
ARG PIXI_ENV=cpu

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS build
ARG PIXI_ENV

WORKDIR /app


COPY . .

RUN pixi install --locked -e ${PIXI_ENV}

# install R's XML/ncdf4 instead of compiling from source
RUN pixi add -e ${PIXI_ENV} r-ncdf4 netcdf4 r-xml libxml2

# system libraries needed by the R packages of IonShed, installed before the environment enters PATH and LD_LIBRARY_PATH
# (apt-get fails with the libraries of the environment); pak then does not install system requirements itself
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcurl4-openssl-dev libssl-dev cmake make libuv1-dev libglpk-dev \
       libxml2-dev pandoc libicu-dev libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PKG_SYSREQS=false

# change paths to force pixi to use the correct environment
ENV PATH=/app/.pixi/envs/${PIXI_ENV}/bin:$PATH
ENV PKG_CONFIG_PATH=/app/.pixi/envs/${PIXI_ENV}/lib/pkgconfig
ENV CPATH=/app/.pixi/envs/${PIXI_ENV}/include
ENV LIBRARY_PATH=/app/.pixi/envs/${PIXI_ENV}/lib
ENV LD_LIBRARY_PATH=/app/.pixi/envs/${PIXI_ENV}/lib

RUN pixi run -e ${PIXI_ENV} install-ionshed

# Python bytecode compiled once here (hash-checked: it stays valid whatever the file dates become in the final image),
# instead of at every run (about 8 s per script otherwise); a few test files of the packages do not compile, hence || true
RUN python -m compileall -q -j 0 --invalidation-mode checked-hash /app/.pixi/envs/${PIXI_ENV}/lib/python3* /app/src /app/scripts || true

RUN pixi shell-hook -e ${PIXI_ENV} -s bash > /shell-hook \
    && echo "#!/bin/bash" > /app/entrypoint.sh \
    && cat /shell-hook >> /app/entrypoint.sh \
    && echo 'exec "$@"' >> /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

FROM ubuntu:24.04 AS production
ARG PIXI_ENV

WORKDIR /app

COPY --from=build /app/.pixi/envs/${PIXI_ENV} /app/.pixi/envs/${PIXI_ENV}
COPY --from=build /app/entrypoint.sh /app/entrypoint.sh
COPY --from=build /app /app

ENV PATH=/app/.pixi/envs/${PIXI_ENV}/bin:$PATH
ENV PKG_CONFIG_PATH=/app/.pixi/envs/${PIXI_ENV}/lib/pkgconfig
ENV CPATH=/app/.pixi/envs/${PIXI_ENV}/include
ENV LIBRARY_PATH=/app/.pixi/envs/${PIXI_ENV}/lib
ENV LD_LIBRARY_PATH=/app/.pixi/envs/${PIXI_ENV}/lib

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-c", "print('massgat environment ready')"]