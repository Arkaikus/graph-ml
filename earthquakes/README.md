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

if nvidia cuda is failing try

```bash
sudo modprobe --remove nvidia_uvm
sudo modprobe nvidia_uvm
```