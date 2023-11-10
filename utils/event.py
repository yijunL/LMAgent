from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
class Event(BaseModel):
    """
    Event class for each agent.
    """
    start_time: datetime
    # action start time
    duration: int = 1
    # action duration. The default unit is hour.
    target_agent: Optional[List[str]]
    # target agent for chatting.
    action_type: str
    # ('shopping','chatting','posting' or 'idle').
    end_time: Optional[datetime]=None
    # action end time

    """Event for each agent"""
    def __init__(self, **data):
        super().__init__(**data)
        self.end_time = self.start_time + timedelta(hours=self.duration)

        
def update_event(original_event, start_time, duration, target_agent, action_type):
    if action_type == "shopping" or action_type=="posting" or action_type=="webcasting":
        result = Event(start_time=start_time, duration=duration, target_agent=None, action_type=action_type)

    if action_type == "chatting":
        # If the agent is not chatting
        if original_event.action_type == "idle" or original_event.action_type == "posting" or original_event.action_type == "webcasting":
            result = Event(start_time=start_time, duration=duration, target_agent=[target_agent], action_type=action_type)
        # If the agent is chatting
        else:
            # maintain maximum chat time
            end_time = start_time + timedelta(hours=duration)  # current chat end time
            original_end_time = original_event.end_time
            if end_time > original_end_time:
                if target_agent in original_event.target_agent:
                    new_target_agent = original_event.target_agent
                else:
                    new_target_agent = original_event.target_agent + [target_agent]
                result = Event(start_time=start_time, duration=duration, target_agent=new_target_agent, action_type=action_type)
            else:
                result = original_event
                if target_agent not in result.target_agent:
                    result.target_agent.append(target_agent)
    return result


def reset_event(start_time):
    return Event(start_time=start_time, duration=0, target_agent=None, action_type='idle')