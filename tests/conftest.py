from hypothesis import settings

# A profile for quick local dev
settings.register_profile("dev", max_examples=5, deadline=None)

# A profile for heavy stress-testing (e.g., CI or nightly runs)
settings.register_profile("stress", max_examples=100, deadline=None)

# Set a default
settings.load_profile("dev")