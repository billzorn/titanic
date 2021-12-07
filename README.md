# Titanic

Titanic is a tool for designing and experimenting
with novel computer arithmetic formats.
It builds on the GNU MPFR library for arbitrary-precision arithmetic,
providing additional support for low-level floating-point behaviors
such as precise control of rounding and precision tracking.

Titanic uses the [FPCore](http://fpbench.org/spec/)
benchmark format for floating-point computations.
Reference interpereters are provided for the FPCore language
using both [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)
and [posit](https://posithub.org/index) arithmetics,
as well as fixed-point.
These interpreters can also interface with each other
(and with custom user-defined arithmetics)
to perform multiple precision, multiple format computations.

## Dependencies

A quick list of what you need to get this running on a Ubuntu 20.04 system:

- antlr4.9, and a java runtime for it (default-jdk works)
- make gcc g++
- python3 python3-venv python3-dev
- libgmp-dev libmpfr-dev libmpc-dev
- npm and nodejs for the webtool

## Setup guide

Here's a list of steps to "install" the tools and make sure they are working.

### OS dependencies

Usually I use more modern Ubuntu LTS releases (16.04 - 20.04). The following should install all of the package-manager level dependencies needed. On other systems, the commands might vary.

`sudo apt install make gcc g++ python3 python3-venv python3-dev libgmp-dev libmpfr-dev libmpc-dev default-jdk`

A recent-ish version of node (16+) is required for the webdemo, unfortunately this does not just `sudo apt install` on Ubuntu 20.04 (you get node 10). The easiest way is probably to stick the prebuilt binaries in some local folder and just invoke npm from there.

### Clone the repo somewhere

`git clone git@github.com:billzorn/titanic.git`

or, for https (i.e. for read only access)

`git clone https://github.com/billzorn/titanic.git`

### Make a Python3 virtual environment

Hopefully Python3.8 is around, so we can use that. Python3.7 also works fine.

```
cd titanic/
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

### Set up the ANTLR4 grammars for FPCore

I use the ANTLR4 tool to generate lexers and parsers for the FPCore language. The generated code is Python, but the generator itself (the ANTLR tool) is Java. First we need to download the latest ANTLR version:

```
cd titanic/titanfp/fpbench/antlr
wget "https://www.antlr.org/download/antlr-4.9.1-complete.jar"
```

Now we can run it, to make the generated Python files that Titanic will expect to exist to provide the lexer and parser:

`make clean && make`

At this point, Titanic should be working. We can test it easily by seeing if the webtool works.

### Set up the Webtool

The Webtool (or webdemo, as the repo calls it) is a big webpack thing that depends on a NodeJS package for deployment. First we need to install the node package:

```
cd titanic/titanfp/web
npm install
```

This might spit out a lot of stuff and complain, but hopefully it works.

### Launch the webtool

Now we can enter the arcane build command to bring up the webtool server. Run this from the root of the Titanic repo (i.e. `titanic/`), and make sure the virtual environment is active (i.e. this shell has run `.env/bin/activate`).

`(cd titanfp/web && npm run build-debug) && python -m titanfp.web.webdemo --serve titanfp/web/dist --host "" --port 8009`

This will attempt to host the webdemo on port 8009; you can visit it with a browser to see if it's working.

Click "Check out the Titanic Evaluator webtool" at the bottom: a split screen interface should appear, with some code on the right and controls on the left. Type any number (i.e. 1.3) into the FPCore arguments box, and click Evalue FPCore underneath: you should see back the same value in the output log, and probably something in the termanil where the webserver is running.

Stop the sever by closing stdin (Ctrl-d) in the terminal.
