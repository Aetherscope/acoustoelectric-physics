# Acoustoelectric interaction fields in electrolyte solutions
[Interuniversity Microelectronics Centre (imec)]: https://www.imec-int.com
[Neuro-electronics Research Flanders (NERF)]: http://nerf.be

Christopher Chare<sup>1</sup>, Rachid Haouari<sup>1</sup>, Klara Volckaert<sup>1</sup>, Agknaton Bottenberg<sup>1</sup>, Sebastian Haesler<sup>2</sup>, Nick Van Helleputte<sup>1</sup>, Xavier Rottenberg<sup>1</sup>, Samer Houri<sup>1</sup>, and Peter Peumans<sup>1</sup>

<sup>1</sup>[Interuniversity Microelectronics Centre (imec)], Belgium\
<sup>2</sup>[Neuro-electronics Research Flanders (NERF)], Belgium

This repository contains the code and data used to reproduce and generate the figures and analysis presented in our paper [link to be updated], submitted to [link to be updated].

## Abstract

The acoustoelectric effect in electrolyte solutions arises from the modulation of local conductivity under variations in applied acoustic pressure. This phenomenon enables non-invasive imaging of (bio-)electrical activity with high spatiotemporal resolution, whereby an ultrasonic focus acts as a virtual probe, upconverting locally induced electrical currents into measurable signals at the ultrasonic frequency. However, a recently developed ab initio formulation of the acoustoelectric model exhibited discrepancies with the long established, tractable acoustoelectric lead-field expression, particularly in the characteristics of the generated acoustoelectric potential field. Here, we experimentally validate the newer ab initio formulation by demonstrating quantitative agreement between simulations and direct field measurements in physiological saline. We further derive a revised lead-field expression from first principles by incorporating the non-trivial acoustoelectric source term described by the ab initio model. Experimental and numerical results from acoustoelectric phantoms confirm the validity of the revised lead-field expression. Our findings establish a unified framework for interpreting and designing emerging acoustoelectric imaging systems, highlighting the importance of applying the appropriate modeling conditions to both distinguish acoustoelectric signals from confounding phenomena and accurately map the underlying current density fields.

## Requirements
  * Python 3.10 or higher
  * See requirements.txt

## Usage
1. Download the simulation and measurement data from FigShare [link to be updated] into their respective folders.
2. Install the necessary libraries with `pip install -r requirements.txt`
3. Run `python ./acoustoelectric_figures.py` to reproduce the results of the acoustoelectric experiments and interaction coefficient analysis.
4. Run `python ./quadrupole_figures.py` to reproduce the results of the simulated arbitrary quadrupole acoustoelectric potential fields.
5. Run `python ./electrokinetic_figures.py` to reproduce the results of the artifacts observed from the acoustoelectric interaction phantom experiments.

## Contact
Corresponding authors:
* Christopher Chare (christopher.chare@imec.be)
* Samer Houri (Samer.Houri@imec.be)

## License
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
