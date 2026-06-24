def __getattr__(name):
    if name == "VisionSignal" or name == "AudioSignal" or name == "TriggerEvent":
        from .signals import VisionSignal, AudioSignal, TriggerEvent
        return locals()[name]
    if name == "StateMachine" or name == "EngineState":
        from .state_machine import StateMachine, EngineState
        return locals()[name]
    if name == "Debounce":
        from .debounce import Debounce
        return Debounce
    if name == "Cooldown":
        from .cooldown import Cooldown
        return Cooldown
    if name == "MappingEngine":
        from .mapping_engine import MappingEngine
        return MappingEngine
    if name == "GestureStabilizer" or name == "GestureEvent" or name == "FeatureState":
        from .gesture_stabilizer import GestureStabilizer, GestureEvent, FeatureState
        return locals()[name]
    if name == "AudioSlot" or name == "AudioState":
        from .audio_slot import AudioSlot, AudioState
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
