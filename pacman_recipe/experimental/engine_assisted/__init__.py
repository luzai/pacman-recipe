"""Opt-in experiment components, independent of the default recipe.

Training/curriculum contracts use the standard library. Import language_lora or
choice_loss explicitly when PyTorch is available; importing this package does
not import PyTorch, models, games or any deployment infrastructure.
"""
