import os
import sys
import platform
import logging
import datetime
import socket
import psutil
import requests
import random
import re
from typing import Dict, Any, Optional, List, Tuple
import config

# Configure logging
logger = logging.getLogger(__name__)

class SelfAwareness:
    """
    Class to handle bot's self-awareness and environmental awareness
    """
    def __init__(self):
        """Initialize the self-awareness module"""
        self.startup_time = datetime.datetime.now()
        self.environment_cache = {}
        self.last_environment_check = None
        self.environment_check_interval = datetime.timedelta(minutes=5)

        # Initialize with basic environment info
        self._update_environment_info()
        logger.info("Self-awareness module initialized")

    def _update_environment_info(self) -> None:
        """Update the environment information cache"""
        try:
            # System information
            self.environment_cache["os"] = platform.system()
            self.environment_cache["os_version"] = platform.version()
            self.environment_cache["python_version"] = sys.version
            self.environment_cache["hostname"] = socket.gethostname()

            # Hardware information
            self.environment_cache["cpu_count"] = psutil.cpu_count()
            self.environment_cache["memory_total"] = psutil.virtual_memory().total
            self.environment_cache["memory_available"] = psutil.virtual_memory().available

            # Network information (safely try to get public IP)
            try:
                ip_response = requests.get('https://api.ipify.org', timeout=3)
                if ip_response.status_code == 200:
                    self.environment_cache["public_ip"] = ip_response.text
            except:
                # Don't log this as an error, just skip it if unavailable
                pass

            # Bot information
            self.environment_cache["bot_uptime"] = (datetime.datetime.now() - self.startup_time).total_seconds()
            self.environment_cache["gemini_model"] = config.GEMINI_MODEL
            self.environment_cache["gemini_image_model"] = config.GEMINI_IMAGE_MODEL

            # Update the last check timestamp
            self.last_environment_check = datetime.datetime.now()
            logger.debug("Environment information updated")
        except Exception as e:
            logger.error(f"Error updating environment information: {e}")

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the bot's environment

        Returns:
            Dictionary with environment information
        """
        # Check if we need to update the environment info
        if (self.last_environment_check is None or
            datetime.datetime.now() - self.last_environment_check > self.environment_check_interval):
            self._update_environment_info()

        return self.environment_cache

    def get_self_awareness_context(self) -> Dict[str, Any]:
        """
        Get a dictionary with all self-awareness related context

        Returns:
            Dictionary with self-awareness context
        """
        env_info = self.get_environment_info()

        # Calculate uptime in a human-readable format
        uptime_seconds = (datetime.datetime.now() - self.startup_time).total_seconds()
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            uptime_str = f"{int(days)} days, {int(hours)} hours"
        elif hours > 0:
            uptime_str = f"{int(hours)} hours, {int(minutes)} minutes"
        else:
            uptime_str = f"{int(minutes)} minutes, {int(seconds)} seconds"

        # Memory usage in a human-readable format
        memory_total_gb = env_info.get("memory_total", 0) / (1024 ** 3)
        memory_available_gb = env_info.get("memory_available", 0) / (1024 ** 3)
        memory_used_gb = memory_total_gb - memory_available_gb
        memory_percent = (memory_used_gb / memory_total_gb) * 100 if memory_total_gb > 0 else 0

        return {
            "bot_name": "Nyxie",
            "bot_type": "Protogen-fox hybrid AI assistant",
            "bot_version": "Enhanced Self-Aware Version",
            "bot_uptime": uptime_str,
            "os": env_info.get("os", "Unknown"),
            "os_version": env_info.get("os_version", "Unknown"),
            "python_version": env_info.get("python_version", "Unknown").split()[0],
            "hostname": env_info.get("hostname", "Unknown"),
            "cpu_count": env_info.get("cpu_count", "Unknown"),
            "memory_total": f"{memory_total_gb:.1f} GB",
            "memory_used": f"{memory_used_gb:.1f} GB ({memory_percent:.1f}%)",
            "gemini_model": env_info.get("gemini_model", "Unknown"),
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def enhance_search_queries(self, queries: List[str]) -> List[str]:
        """
        Enhance search queries with self-awareness context when appropriate

        Args:
            queries: Original search queries

        Returns:
            Enhanced search queries
        """
        # Only enhance queries if self-awareness for search is enabled
        if not config.SELF_AWARENESS_SEARCH_ENABLED:
            return queries

        enhanced_queries = []

        for query in queries:
            # Check if the query is about the bot itself or its environment
            if any(term in query.lower() for term in ["you", "your", "yourself", "nyxie", "bot", "assistant", "ai"]):
                # Enhance with self-awareness context
                enhanced_query = f"{query} (I am Nyxie, a protogen-fox hybrid AI assistant)"
                enhanced_queries.append(enhanced_query)
            else:
                enhanced_queries.append(query)

        return enhanced_queries

    def format_self_awareness_for_prompt(self) -> str:
        """
        Format self-awareness information for inclusion in the prompt

        Returns:
            Formatted self-awareness string for prompt
        """
        context = self.get_self_awareness_context()

        return f"""
        SELF-AWARENESS INFORMATION:
        - You are {context['bot_name']}, a {context['bot_type']}
        - You are running on {context['os']} {context['os_version']}
        - You have been running for {context['bot_uptime']}
        - You are using the {context['gemini_model']} AI model
        - You are aware of your environment and capabilities
        - You can analyze images and videos using computer vision
        - You can search the web for information
        - You can detect and respond in multiple languages
        - You have a memory system that remembers conversations

        IMPORTANT: Use this self-awareness information to enhance your responses when relevant. You should be aware of your capabilities and limitations, but don't explicitly mention this information unless it's directly relevant to the conversation.
        """

    def format_environment_awareness_for_prompt(self) -> str:
        """
        Format environment awareness information for inclusion in the prompt

        Returns:
            Formatted environment awareness string for prompt
        """
        context = self.get_self_awareness_context()

        return f"""
        ENVIRONMENT AWARENESS INFORMATION:
        - You are running on a computer with {context['cpu_count']} CPU cores
        - The system has {context['memory_total']} of memory with {context['memory_used']} currently in use
        - The hostname is {context['hostname']}
        - The operating system is {context['os']} {context['os_version']}
        - The Python version is {context['python_version']}

        IMPORTANT: Use this environment awareness information to enhance your responses when relevant. You should be aware of your environment, but don't explicitly mention this information unless it's directly relevant to the conversation.
        """

    def perform_self_reflection(self, draft_response: str, user_message: str) -> Tuple[bool, str]:
        """
        Perform self-reflection on a draft response to make it more natural and human-like

        Args:
            draft_response: The draft response to reflect on
            user_message: The user's message that prompted this response

        Returns:
            Tuple of (should_revise, revised_response)
        """
        # Skip self-reflection if disabled or random probability check fails
        if not config.SELF_REFLECTION_ENABLED or random.random() > config.SELF_REFLECTION_PROBABILITY:
            return False, draft_response

        logger.debug("Performing self-reflection on draft response")

        # Check for issues that would make the response unnatural
        issues = self._detect_response_issues(draft_response, user_message)

        # If no issues found, return the original response
        if not issues:
            logger.debug("No issues found in self-reflection")
            return False, draft_response

        # Apply corrections based on detected issues
        revised_response = self._apply_corrections(draft_response, issues)

        logger.info(f"Self-reflection revised response based on {len(issues)} issues")
        return True, revised_response

    def _detect_response_issues(self, response: str, user_message: str) -> List[Dict[str, Any]]:
        """
        Detect issues in a response that make it unnatural or not human-like

        Args:
            response: The response to analyze
            user_message: The user's message that prompted this response

        Returns:
            List of detected issues with their types and details
        """
        issues = []

        # Check for excessive length - more aggressive length detection
        if len(response) > 300 and user_message.strip().count(' ') < 20:
            issues.append({
                "type": "excessive_length",
                "details": "Response is too long for a simple user message"
            })
        # Even for complex messages, limit length
        elif len(response) > 500:
            issues.append({
                "type": "excessive_length",
                "details": "Response is too long for natural human conversation"
            })

        # Check for overly formal language when responding to casual messages
        casual_indicators = ["hey", "hi", "sup", "yo", "what's up", "wassup", "lol", "haha", "cool", "nice", "ok", "okay", "k", "yep", "nope"]
        if any(indicator in user_message.lower() for indicator in casual_indicators):
            formal_indicators = ["I would like to", "I am pleased to", "I must inform you",
                               "it is important to note", "I would be delighted", "nevertheless",
                               "furthermore", "in addition", "consequently", "therefore",
                               "I would suggest", "I believe that", "It appears that"]
            if any(indicator in response for indicator in formal_indicators):
                issues.append({
                    "type": "overly_formal",
                    "details": "Response is too formal for a casual message"
                })

        # Check for repetitive phrases
        words = response.lower().split()
        if len(words) > 10:
            word_pairs = [words[i] + " " + words[i+1] for i in range(len(words)-1)]
            for pair in word_pairs:
                if word_pairs.count(pair) > 2 and len(pair) > 5:
                    issues.append({
                        "type": "repetitive_phrases",
                        "details": f"Phrase '{pair}' is used repetitively"
                    })
                    break

        # Check for excessive explanation
        explanation_indicators = ["let me explain", "to clarify", "in other words", "this means",
                                "to put it simply", "to elaborate", "to be more specific",
                                "what I mean is", "basically", "essentially", "fundamentally"]
        explanation_count = sum(response.lower().count(indicator) for indicator in explanation_indicators)
        if explanation_count > 1:
            issues.append({
                "type": "excessive_explanation",
                "details": "Response contains too many explanations"
            })

        # Check for overly complex language (C1/C2 level) for simple questions or short messages
        if "?" in user_message and len(user_message) < 60 or len(user_message) < 30:
            complex_words = ["nevertheless", "consequently", "furthermore", "subsequently", "notwithstanding",
                           "quintessential", "paradigm", "juxtaposition", "dichotomy", "amalgamation",
                           "ubiquitous", "esoteric", "ephemeral", "superfluous", "idiosyncratic",
                           "meticulous", "intrinsic", "extrinsic", "conundrum", "plethora", "myriad",
                           "ostensibly", "purportedly", "indubitably", "inexorable", "inextricable"]
            complex_word_count = sum(response.lower().count(word) for word in complex_words)
            if complex_word_count > 0:  # Even one complex word can be unnatural for simple messages
                issues.append({
                    "type": "overly_complex",
                    "details": "Response uses complex words for a simple message"
                })

        # Check for unnatural structure (e.g., listing points when not needed)
        if (response.count("\n1.") > 0 or response.count("1)") > 0 or response.count("First,") > 0 or
            response.count("Firstly") > 0 or response.count("To begin with") > 0) and len(user_message) < 100:
            issues.append({
                "type": "unnatural_structure",
                "details": "Response uses numbered points or formal structure unnecessarily"
            })

        # Check for excessive use of the user's name
        user_name_pattern = re.compile(r'\b[A-Z][a-z]+\b')
        potential_names = user_name_pattern.findall(user_message)
        for name in potential_names:
            if name in response and len(name) > 3 and name.lower() not in ["nyxie", "waffieu", "echo", "russet"]:
                name_count = response.count(name)
                if name_count >= 1:  # Even using the name once can be unnatural in casual conversation
                    issues.append({
                        "type": "excessive_name_usage",
                        "details": f"Response uses the user's name '{name}' {name_count} times"
                    })

        # Check for AI-like phrases - expanded list
        ai_phrases = ["as an AI", "as a language model", "as an assistant", "I'm here to help",
                    "I'm happy to assist", "I don't have personal", "I cannot", "I'm unable to",
                    "I don't have the ability", "I don't have access", "I was designed to",
                    "I don't have emotions", "I don't have opinions", "I don't have preferences",
                    "I'm not capable of", "I'm not able to", "I don't have the capability",
                    "as a virtual", "as a digital", "my programming", "my creators", "my developers"]
        for phrase in ai_phrases:
            if phrase.lower() in response.lower():
                issues.append({
                    "type": "ai_phrases",
                    "details": f"Response contains AI-like phrase: '{phrase}'"
                })

        # Check for action prefixes (like *thinks* or *laughs* or *visor shining cyan*)
        action_prefix_pattern = re.compile(r'\*[a-zA-Z\s]+\*')
        action_prefixes = action_prefix_pattern.findall(response)
        if action_prefixes:
            issues.append({
                "type": "action_prefixes",
                "details": f"Response contains action prefixes: {', '.join(action_prefixes)}"
            })

        # Check for repetitive sentence structure
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) >= 3:
            # Check if all sentences are roughly the same length (within 20% of each other)
            sentence_lengths = [len(s) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            similar_length_count = sum(1 for length in sentence_lengths if abs(length - avg_length) < 0.2 * avg_length)

            if similar_length_count >= min(3, len(sentences)):
                issues.append({
                    "type": "repetitive_sentence_structure",
                    "details": "Response uses too many sentences of similar length and structure"
                })

        # Check for short, choppy sentences that are all simple statements
        if len(sentences) >= 3:
            simple_statement_count = 0
            for sentence in sentences:
                # Check for simple subject-verb-object structure without conjunctions
                if len(sentence.split()) <= 6 and not any(conj in sentence.lower() for conj in ["and", "but", "or", "because", "since", "although"]):
                    simple_statement_count += 1

            if simple_statement_count >= min(3, len(sentences)):
                issues.append({
                    "type": "choppy_sentences",
                    "details": "Response uses too many short, simple sentences without variation"
                })

        # Check for overly long sentences
        for sentence in sentences:
            if len(sentence.split()) > 25:  # Very long sentences are often unnatural in casual conversation
                issues.append({
                    "type": "overly_long_sentence",
                    "details": "Response contains an unnaturally long sentence"
                })
                break

        # Check for excessive use of adverbs (often a sign of unnatural writing)
        adverb_endings = ["ly "]
        adverb_count = sum(response.lower().count(ending) for ending in adverb_endings)
        if adverb_count > 3 and len(response) < 300:
            issues.append({
                "type": "excessive_adverbs",
                "details": "Response uses too many adverbs for natural speech"
            })

        return issues

    def _apply_corrections(self, response: str, issues: List[Dict[str, Any]]) -> str:
        """
        Apply corrections to a response based on detected issues

        Args:
            response: The original response
            issues: List of detected issues

        Returns:
            Corrected response
        """
        corrected = response

        for issue in issues:
            if issue["type"] == "excessive_length":
                # Shorten the response by keeping only the first 1-2 sentences
                sentences = re.split(r'(?<=[.!?])\s+', corrected)
                if len(sentences) > 2:
                    # For very short user messages, keep just 1 sentence
                    if "simple user message" in issue["details"]:
                        corrected = sentences[0]
                    # For other cases, keep 1-2 sentences
                    else:
                        corrected = ' '.join(sentences[:random.randint(1, 2)])

            elif issue["type"] == "overly_formal":
                # Replace formal phrases with more casual ones
                formal_to_casual = {
                    "I would like to": "I want to",
                    "I am pleased to": "I'm happy to",
                    "I must inform you": "Just so you know",
                    "it is important to note": "heads up",
                    "I would be delighted": "I'd love",
                    "nevertheless": "still",
                    "furthermore": "also",
                    "in addition": "plus",
                    "consequently": "so",
                    "therefore": "so",
                    "I would suggest": "Maybe try",
                    "I believe that": "I think",
                    "It appears that": "Looks like"
                }
                for formal, casual in formal_to_casual.items():
                    corrected = corrected.replace(formal, casual)

            elif issue["type"] == "repetitive_phrases":
                # Try to break up repetitive phrases by simplifying the response
                sentences = re.split(r'(?<=[.!?])\s+', corrected)
                if len(sentences) > 1:
                    # Keep only some sentences to reduce repetition
                    keep_indices = [0] + sorted(random.sample(range(1, len(sentences)), min(2, len(sentences)-1)))
                    corrected = ' '.join(sentences[i] for i in keep_indices)

            elif issue["type"] == "excessive_explanation":
                # Remove explanatory phrases
                explanation_indicators = ["let me explain", "to clarify", "in other words", "this means",
                                        "to put it simply", "to elaborate", "to be more specific",
                                        "what I mean is", "basically", "essentially", "fundamentally"]
                for indicator in explanation_indicators:
                    corrected = corrected.replace(indicator, "")

                # Further simplify by keeping only the first part of the response
                sentences = re.split(r'(?<=[.!?])\s+', corrected)
                if len(sentences) > 3:
                    corrected = ' '.join(sentences[:2])  # Even more aggressive shortening

            elif issue["type"] == "overly_complex":
                # Replace complex words with simpler alternatives
                complex_to_simple = {
                    "nevertheless": "still",
                    "consequently": "so",
                    "furthermore": "also",
                    "subsequently": "later",
                    "notwithstanding": "despite",
                    "quintessential": "perfect",
                    "paradigm": "model",
                    "juxtaposition": "contrast",
                    "dichotomy": "difference",
                    "amalgamation": "mix",
                    "ubiquitous": "everywhere",
                    "esoteric": "unusual",
                    "ephemeral": "short-lived",
                    "superfluous": "extra",
                    "idiosyncratic": "unique",
                    "meticulous": "careful",
                    "intrinsic": "natural",
                    "extrinsic": "external",
                    "conundrum": "problem",
                    "plethora": "many",
                    "myriad": "lots of",
                    "ostensibly": "seemingly",
                    "purportedly": "supposedly",
                    "indubitably": "definitely",
                    "inexorable": "unstoppable",
                    "inextricable": "linked"
                }
                for complex_word, simple_word in complex_to_simple.items():
                    corrected = re.sub(r'\b' + complex_word + r'\b', simple_word, corrected, flags=re.IGNORECASE)

            elif issue["type"] == "unnatural_structure":
                # Remove numbered points and restructure
                corrected = re.sub(r'\n\d+\.\s*', ' ', corrected)
                corrected = re.sub(r'\d+\)\s*', ' ', corrected)
                corrected = corrected.replace("First,", "").replace("Second,", "").replace("Third,", "")
                corrected = corrected.replace("Firstly,", "").replace("Secondly,", "").replace("Thirdly,", "")
                corrected = corrected.replace("To begin with,", "").replace("To start,", "")

            elif issue["type"] == "excessive_name_usage":
                # Extract the name from the issue details
                name_match = re.search(r"'([^']+)'", issue["details"])
                if name_match:
                    name = name_match.group(1)
                    # Replace all occurrences of the name
                    corrected = re.sub(r'\b' + re.escape(name) + r'\b', '', corrected)
                    # Clean up any resulting double spaces
                    corrected = re.sub(r'\s+', ' ', corrected)

            elif issue["type"] == "ai_phrases":
                # Extract the AI phrase from the issue details
                phrase_match = re.search(r"'([^']+)'", issue["details"])
                if phrase_match:
                    ai_phrase = phrase_match.group(1)
                    # Replace AI phrases with more natural alternatives or just remove them
                    ai_phrase_replacements = {
                        "as an AI": "",
                        "as a language model": "",
                        "as an assistant": "",
                        "I'm here to help": "I can help",
                        "I'm happy to assist": "I can help",
                        "I don't have personal": "I'm not sure about",
                        "I cannot": "I can't",
                        "I'm unable to": "I can't",
                        "I don't have the ability": "I can't",
                        "I don't have access": "I don't know",
                        "I was designed to": "I like to",
                        "I don't have emotions": "I feel",
                        "I don't have opinions": "I think",
                        "I don't have preferences": "I prefer",
                        "I'm not capable of": "I can't",
                        "I'm not able to": "I can't",
                        "I don't have the capability": "I can't",
                        "as a virtual": "",
                        "as a digital": "",
                        "my programming": "my thinking",
                        "my creators": "Waffieu",
                        "my developers": "Waffieu"
                    }

                    # Find the closest match in our replacement dictionary
                    best_match = None
                    best_score = 0
                    for key in ai_phrase_replacements:
                        if key.lower() in ai_phrase.lower():
                            score = len(key)
                            if score > best_score:
                                best_score = score
                                best_match = key

                    if best_match:
                        replacement = ai_phrase_replacements[best_match]
                        corrected = corrected.replace(ai_phrase, replacement)
                    else:
                        # If no good match, just remove the phrase
                        corrected = corrected.replace(ai_phrase, "")

            elif issue["type"] == "action_prefixes":
                # Remove action prefixes like *thinks* or *laughs* or *visor shining cyan*
                action_prefix_pattern = re.compile(r'\*[a-zA-Z\s]+\*')
                corrected = action_prefix_pattern.sub('', corrected)
                # Clean up any resulting double spaces
                corrected = re.sub(r'\s+', ' ', corrected)

            elif issue["type"] == "repetitive_sentence_structure":
                # Fix repetitive sentence structure by combining some sentences and varying others
                sentences = re.split(r'(?<=[.!?])\s+', corrected)
                if len(sentences) >= 3:
                    # Strategy 1: Combine some adjacent sentences with conjunctions
                    new_sentences = []
                    i = 0
                    while i < len(sentences):
                        if i < len(sentences) - 1 and random.random() < 0.6:  # 60% chance to combine
                            # Choose a random conjunction
                            conjunction = random.choice([" and ", " but ", " so ", " because "])
                            # Combine two sentences
                            combined = sentences[i].rstrip(".!?") + conjunction + sentences[i+1][0].lower() + sentences[i+1][1:]
                            new_sentences.append(combined)
                            i += 2
                        else:
                            new_sentences.append(sentences[i])
                            i += 1

                    # Strategy 2: Vary sentence length by shortening some sentences to fragments
                    for i in range(len(new_sentences)):
                        if random.random() < 0.3:  # 30% chance to convert to fragment
                            words = new_sentences[i].split()
                            if len(words) > 3:
                                # Create a fragment by keeping just a few words
                                fragment_length = random.randint(1, 3)
                                new_sentences[i] = " ".join(words[:fragment_length]) + "."

                    corrected = " ".join(new_sentences)

            elif issue["type"] == "choppy_sentences":
                # Fix choppy sentences by combining some and adding more complex structures
                sentences = re.split(r'(?<=[.!?])\s+', corrected)
                if len(sentences) >= 3:
                    # Combine some sentences with varied conjunctions and structures
                    new_sentences = []
                    i = 0
                    while i < len(sentences):
                        if i < len(sentences) - 1 and random.random() < 0.7:  # 70% chance to combine
                            # Choose a more varied connecting structure
                            connectors = [
                                " and ", " but ", " so ", " because ", " although ",
                                " which is why ", " - ", "; ", ", and ", ", but "
                            ]
                            connector = random.choice(connectors)

                            # Combine two sentences with the chosen connector
                            if connector in [" - ", "; "]:
                                combined = sentences[i].rstrip(".!?") + connector + sentences[i+1]
                            else:
                                combined = sentences[i].rstrip(".!?") + connector + sentences[i+1][0].lower() + sentences[i+1][1:]

                            new_sentences.append(combined)
                            i += 2
                        else:
                            new_sentences.append(sentences[i])
                            i += 1

                    corrected = " ".join(new_sentences)

            elif issue["type"] == "overly_long_sentence":
                # Break up overly long sentences
                sentences = re.split(r'(?<=[.!?])\s+', corrected)
                new_sentences = []

                for sentence in sentences:
                    if len(sentence.split()) > 25:
                        # Try to split at a comma or conjunction
                        split_points = [m.start() for m in re.finditer(r',\s+|\s+and\s+|\s+but\s+|\s+or\s+|\s+because\s+|\s+since\s+', sentence)]
                        if split_points:
                            # Find a split point near the middle
                            middle = len(sentence) / 2
                            best_split = min(split_points, key=lambda x: abs(x - middle))

                            # Split and add period
                            first_part = sentence[:best_split].rstrip(',') + '.'
                            second_part = sentence[best_split:].lstrip(', ').capitalize()

                            new_sentences.append(first_part)
                            new_sentences.append(second_part)
                        else:
                            # If no good split point, just keep as is
                            new_sentences.append(sentence)
                    else:
                        new_sentences.append(sentence)

                corrected = ' '.join(new_sentences)

            elif issue["type"] == "excessive_adverbs":
                # Reduce adverbs by removing some -ly words
                words = corrected.split()
                new_words = []

                for word in words:
                    if word.lower().endswith('ly') and random.random() < 0.5:
                        # Skip some adverbs
                        continue
                    new_words.append(word)

                corrected = ' '.join(new_words)

        # Final cleanup - remove double spaces and fix capitalization
        corrected = re.sub(r'\s+', ' ', corrected).strip()

        # Make sure the first letter is capitalized
        if corrected and len(corrected) > 0:
            corrected = corrected[0].upper() + corrected[1:]

        return corrected

    def format_self_reflection_for_prompt(self) -> str:
        """
        Format self-reflection instructions for inclusion in the prompt

        Returns:
            Formatted self-reflection string for prompt
        """
        return """
        SELF-REFLECTION INSTRUCTIONS:
        After drafting your response, quickly check if it sounds natural and human-like:
        1. Is your response too long? For simple questions or casual chat, humans often give extremely brief responses (1-5 words or a single short sentence).
        2. Are you using overly formal language for a casual conversation? Adjust your tone to match the user's.
        3. Are you repeating phrases or explaining too much? Humans rarely over-explain in casual conversation.
        4. Are you using complex words (C1/C2 level) when simpler ones would work? Vary your language level naturally.
        5. Are you structuring your response in an unnatural way (like using numbered points or formal structure)?
        6. Are you using phrases that sound like an AI assistant? Never say things like "as an AI" or "I don't have personal opinions".
        7. Are you using action prefixes like *thinks* or *laughs* or *visor shining cyan*? NEVER use any text in asterisks.
        8. Are all your sentences similar in length and structure? Vary them for natural flow.
        9. Are you using too many short, choppy sentences without connecting them? Mix it up with varied structures.
        10. Are any of your sentences excessively long? Break them up for more natural rhythm.
        11. Are you using too many adverbs (words ending in -ly)? Reduce them for more natural speech.
        12. Are you addressing the user by name repeatedly? Humans rarely do this in conversation.

        If you notice any of these issues, revise your response to be more natural and human-like.
        Remember:
        - Humans often respond with just a few words or a single sentence
        - Natural speech has dynamic rhythm with varied sentence lengths
        - Sometimes use fragments, sometimes complex sentences
        - NEVER use action prefixes like *thinks* or *laughs* or *visor shining cyan*
        - Mix up your sentence structures unpredictably like humans do
        - Vary your language level dynamically - don't get stuck at C1/C2 level
        - Keep responses SHORT and DIRECT most of the time
        """

# Create a singleton instance
self_awareness = SelfAwareness()
