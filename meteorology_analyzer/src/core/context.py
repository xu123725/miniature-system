from typing import Dict, List, Any, Optional
import time
import logging

logger = logging.getLogger("meteorology_analyzer")

class ContextManager:
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.state = {
            "last_site": None,
            "last_time_range": None,
            "last_calculated_field": None,
            "last_visualized_type": None,
            "last_tool_used": None,
            "calculation_results": {},
            "current_turn": 0,
            "entity_history": [],
            "visualization_history": [],
            "conversation_topic": None,
            "context_dependencies": {}
        }

    def update_state(self, **kwargs):
        """Update context state variables."""
        for key, value in kwargs.items():
            if key in self.state:
                self.state[key] = value
                logger.debug(f"Context updated: {key} = {value}")
            else:
                logger.warning(f"Attempted to update unknown state key: {key}")

    def record_interaction(self, user_input: str, system_response: str, 
                          intent: str = None, entities: Dict = None, status: str = "success"):
        """Record a conversation turn."""
        self.state["current_turn"] += 1
        
        user_msg = {
            "role": "user",
            "content": user_input,
            "timestamp": time.time(),
            "intent": intent,
            "entities": entities or {}
        }
        
        sys_msg = {
            "role": "system",
            "content": system_response,
            "timestamp": time.time(),
            "status": status
        }
        
        self.conversation_history.append(user_msg)
        self.conversation_history.append(sys_msg)
        
        # Update entity history
        if entities:
            self.state["entity_history"].append({
                "turn": self.state["current_turn"],
                "timestamp": time.time(),
                "entities": entities.copy()
            })

    def get_context_summary(self) -> str:
        """Generate a summary string of the current context for LLM prompts."""
        summary = "Current Context:\n"
        if self.state["last_site"]:
            summary += f"- Focus Site: {self.state['last_site']}\n"
        if self.state["last_calculated_field"]:
            summary += f"- Last Calculation: {self.state['last_calculated_field']}\n"
        if self.state["last_visualized_type"]:
            summary += f"- Last Chart: {self.state['last_visualized_type']}\n"
        
        if self.conversation_history:
            summary += "\nRecent Conversation:\n"
            for msg in self.conversation_history[-4:]: # Last 2 turns
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                summary += f"{role}: {content}\n"
        
        return summary
