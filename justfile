set dotenv-load

default: build

build target="": (build-desktop target) (build-web target)

desktop target="": (build-desktop target)
web target="": (build-web target)

build-desktop target:
    @echo 'Building for desktop ({{target}})..'
    cargo build {{target}}

build-web target:
    @echo "Building for web ({{target}}).."
    wasm-pack build {{target}}
    rm -rf www/res && cp -rPf res/ www/res
    cd www/ && npm install

run-desktop target="":
    @echo "Running on desktop.."
    cargo run {{target}}

run-web target="": (build-web target)
    @echo "Running on web.."
    cd www/ && npm run start
