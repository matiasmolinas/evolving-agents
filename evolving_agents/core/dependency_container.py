from typing import Dict, Any, Optional

class DependencyContainer:
    """Manages component dependencies and circular references."""
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._initialized = False
        
    def register(self, name: str, component: Any):
        """Register a component."""
        self._components[name] = component
        
    def get(self, name: str) -> Any:
        """Get a registered component."""
        if name not in self._components:
            raise ValueError(f"Component {name} not registered")
        return self._components[name]
    
    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components
    
    async def initialize(self):
        """Complete initialization of all components."""
        if self._initialized:
            return
            
        # Initialize components in proper order if needed
        # This method should be implemented by the application
        
        self._initialized = True