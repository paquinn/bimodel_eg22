# Differentiable 3D CAD Programs for Bidirectional Editing

This repository contains the reference implementation of the paper "Differentiable 3D CAD Programs for Bidirectional Editing", conditionally accepted to Eurographics 2022.

Modern CAD tools represent 3D designs not only as geometry, but also as a program composed of geometric operations, each of which depends on a set of parameters. Program representations enable meaningful and controlled shape variations via parameter changes. However, achieving desired modifications solely through parameter editing is challenging when CAD models have not been explicitly authored to expose select degrees of freedom in advance.

We introduce a novel bidirectional editing system for 3D CAD programs. In addition to editing the CAD program, users can directly manipulate 3D geometry and our system infers parameter updates to keep both representations in sync.

We formulate inverse edits as a set of constrained optimization objectives, returning plausible updates to program parameters that both match user intent and maintain program validity. Our approach implements an automatically differentiable domain-specific language for CAD programs, providing derivatives for this optimization to be performed quickly on any expressed program.

Our system enables rapid, interactive exploration of a constrained 3D design space by allowing users to manipulate the program and geometry interchangeably during design iteration. While our approach is not designed to optimize across changes in geometric topology, we show it is expressive and performant enough for users to produce a diverse set of design variants, even when the CAD program contains a relatively large number of parameters.

## Installation and Setup

Informally, the implementation is an extension meant to run in Blender. It uses a few external python dependencies (notably, Numpy, Scipy, and CFFI) in a Conda virtual environment. It has been built and tested with Blender 2.83 on Mac OS High Sierra and Catalina.

The addon can be imported to blender by importing `__init__.py`.

Detailed setup and dependency instructions coming soon.

## Language

Please see the [Language Reference](/language-reference.md)
