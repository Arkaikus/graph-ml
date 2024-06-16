# earthquakes package

## setup

install this package locally with `pip install -Ie .

## troubleshooting

if `quakes` command is not available add this to `.bashrc`

```bash
# Add ~/.local/bin to PATH
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi

```