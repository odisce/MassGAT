ARG PIXI_VERSION=0.75.0
ARG PIXI_ENV=cpu

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS build
ARG PIXI_ENV

WORKDIR /app


COPY . .

RUN pixi install --locked -e ${PIXI_ENV}

# install R's XML/ncdf4 instead of compiling from source
RUN pixi add -e ${PIXI_ENV} r-ncdf4 netcdf4 r-xml libxml2

# change paths to force pixi to use the correct environment
ENV PATH=/app/.pixi/envs/${PIXI_ENV}/bin:$PATH
ENV PKG_CONFIG_PATH=/app/.pixi/envs/${PIXI_ENV}/lib/pkgconfig
ENV CPATH=/app/.pixi/envs/${PIXI_ENV}/include
ENV LIBRARY_PATH=/app/.pixi/envs/${PIXI_ENV}/lib
ENV LD_LIBRARY_PATH=/app/.pixi/envs/${PIXI_ENV}/lib

RUN pixi run -e ${PIXI_ENV} install-ionshed

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

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-c", "print('massgat environment ready')"]