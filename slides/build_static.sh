#!/usr/bin/env bash
set -e

md="$1"
base=$(basename "$md" .md)

num=${base%%-*}
slug=${base#*-}
title=$(echo "$slug" | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')

final_html="build/${base}.html"

mkdir -p build

# Copy images folder if it exists alongside the markdown file
md_dir=$(dirname "$md")

#node build/render.js prep/foundations/01_orientation/00-intro.md build/template_static.html build/00-intro.html
node build/render.js "$md" build/template_static.html "$final_html" "$title" "$num" "$title"

echo "Built $final_html"
