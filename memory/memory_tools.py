"""Memory-related function calling tools for the Orchestrator.

These tools are registered with the Gemini agent for function calling.
"""

import logging

logger = logging.getLogger(__name__)


def preload_memory(user_id: str, context: str = "") -> dict:
    """Preload relevant long-term memories for the current conversation context.

    This tool is called by the Orchestrator to fetch memories
    that should be injected into the conversation context.

    Args:
        user_id: The user identifier.
        context: Current conversation context or query.

    Returns:
        Dict with memories list, count, and user_id.
    """
    from memory.memory_bank import load_relevant_memories

    memories = load_relevant_memories(user_id, context, top_k=3)
    return {
        "memories": memories,
        "count": len(memories),
        "user_id": user_id,
    }


def forget_recent_memory(user_id: str, minutes: int = 30) -> dict:
    """Forget (delete) memories created within the last N minutes.

    Reserved interface for user-triggered memory deletion.
    Allows users to say "forget what I just told you" to remove
    recent memories from the system.

    Args:
        user_id: The user identifier.
        minutes: How far back to delete (default 30 minutes).

    Returns:
        Dict with deleted count and status.
    """
    from memory.memory_bank import _get_bank

    bank = _get_bank(user_id)
    deleted = bank.delete_recent_memories(minutes=minutes)
    return {
        "deleted": deleted,
        "minutes": minutes,
        "status": "ok",
    }


def forget_memory(user_id: str, memory_id: str) -> dict:
    """Delete a specific memory by its ID.

    Args:
        user_id: The user identifier.
        memory_id: The memory document ID to delete.

    Returns:
        Dict with deletion status.
    """
    from memory.memory_bank import _get_bank

    bank = _get_bank(user_id)
    success = bank.delete_memory(memory_id)
    return {
        "memory_id": memory_id,
        "deleted": success,
        "status": "ok" if success else "not_found",
    }


def remember_entity(user_id: str, name: str, entity_type: str = "person", attributes: str = "") -> dict:
    """Remember a person, place, or thing the user explicitly wants to save.

    Called when user says things like "Remember that David works at the cafe."

    Args:
        user_id: The user identifier.
        name: Name of the entity (person, place, org).
        entity_type: One of "person", "place", "organization", "event".
        attributes: Comma-separated key facts (e.g. "role=coworker,likes=coffee").

    Returns:
        Dict with entity_id, name, and status.
    """
    try:
        from context.entity_graph import Entity, EntityGraphService

        graph = EntityGraphService(user_id)

        # Parse attributes string into dict
        attrs = {}
        if attributes:
            for pair in attributes.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    attrs[k.strip()] = v.strip()
                elif pair:
                    attrs["description"] = pair

        # Check if entity already exists
        existing = graph.find_entity_by_name(name, entity_type=entity_type)
        if existing:
            # Update existing entity
            if attrs:
                merged = {**existing.attributes, **attrs}
                graph.update_entity(existing.entity_id, {"attributes": merged})
            return {
                "entity_id": existing.entity_id,
                "name": name,
                "status": "updated",
                "message": f"Updated existing {entity_type} '{name}'.",
            }

        # Create new entity
        entity = Entity(
            entity_type=entity_type,
            name=name,
            attributes=attrs,
        )
        eid = graph.create_entity(entity)
        if eid:
            return {
                "entity_id": eid,
                "name": name,
                "status": "created",
                "message": f"I'll remember {name}.",
            }
        return {"name": name, "status": "failed", "message": "Could not save entity."}
    except Exception:
        logger.exception("remember_entity failed for user %s", user_id)
        return {"name": name, "status": "error", "message": "Memory system unavailable."}


def what_do_you_remember(user_id: str, query: str = "") -> dict:
    """Tell the user what the system remembers about them or a specific topic.

    Called when user asks "What do you remember about me?" or "about David?"

    Args:
        user_id: The user identifier.
        query: Optional name or topic to focus on.

    Returns:
        Dict with summary text and relevant memories/entities.
    """
    parts = []

    # Fetch relevant memories
    try:
        from memory.memory_bank import load_relevant_memories
        memories = load_relevant_memories(user_id, query or "user profile", top_k=5)
        if memories:
            parts.append("Here's what I remember:")
            for m in memories:
                parts.append(f"- {m}")
    except Exception:
        logger.debug("Memory retrieval failed", exc_info=True)

    # If querying a specific name, check entity graph
    if query:
        try:
            from context.entity_graph import EntityGraphService
            graph = EntityGraphService(user_id)
            entity = graph.find_entity_by_name(query)
            if entity:
                parts.append(f"\nAbout {entity.name} ({entity.entity_type}):")
                for k, v in entity.attributes.items():
                    parts.append(f"- {k}: {v}")
                # Get connected entities
                connected = graph.get_connected_entities(entity.entity_id)
                if connected:
                    names = [e.name for e in connected[:5]]
                    parts.append(f"- Connected to: {', '.join(names)}")
        except Exception:
            logger.debug("Entity graph lookup failed", exc_info=True)

    if not parts:
        summary = "I don't have any specific memories stored yet. As we interact, I'll remember important things you share."
    else:
        summary = "\n".join(parts)

    return {
        "summary": summary,
        "user_id": user_id,
        "query": query,
    }


def forget_entity(user_id: str, name: str) -> dict:
    """Forget a person, place, or thing entirely.

    Deletes the entity from the graph and any memories referencing it.
    Called when user says "Forget about David."

    Args:
        user_id: The user identifier.
        name: Name of the entity to forget.

    Returns:
        Dict with status and confirmation message.
    """
    try:
        from context.entity_graph import EntityGraphService

        graph = EntityGraphService(user_id)
        entity = graph.find_entity_by_name(name)
        if not entity:
            return {
                "name": name,
                "status": "not_found",
                "message": f"I don't have any record of '{name}'.",
            }

        eid = entity.entity_id
        success = graph.delete_entity(eid)

        # Also delete memories referencing this entity
        deleted_memories = 0
        try:
            from memory.memory_bank import _get_bank
            bank = _get_bank(user_id)
            # Retrieve memories that reference this entity
            results = bank.retrieve_memories(name, top_k=10)
            for mem in results:
                entity_refs = mem.get("entity_refs", [])
                if eid in entity_refs:
                    bank.delete_memory(mem.get("memory_id", ""))
                    deleted_memories += 1
        except Exception:
            logger.debug("Failed to clean up entity-related memories", exc_info=True)

        if success:
            return {
                "name": name,
                "entity_id": eid,
                "status": "deleted",
                "deleted_memories": deleted_memories,
                "message": f"I've forgotten about {name}.",
            }
        return {"name": name, "status": "failed", "message": "Could not delete entity."}
    except Exception:
        logger.exception("forget_entity failed for user %s", user_id)
        return {"name": name, "status": "error", "message": "Memory system unavailable."}


MEMORY_FUNCTIONS = {
    "preload_memory": preload_memory,
    "forget_recent_memory": forget_recent_memory,
    "forget_memory": forget_memory,
    "remember_entity": remember_entity,
    "what_do_you_remember": what_do_you_remember,
    "forget_entity": forget_entity,
}
