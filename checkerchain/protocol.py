# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing
import bittensor as bt

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class CheckerChainMinerResponse(typing.NamedTuple):
    score: float | None = None
    review: str | None = None
    keywords: list[str] = []


class CheckerChainSynapse(bt.Synapse):
    """
    A protocol representation for handling request and response communication between
    the miner and the validator, where each response contains a score, review, and keywords.

    Attributes:
    - query: A list of strings representing the input request sent by the validator.
    - response: A list of dictionaries, each with keys 'score' (float), 'review' (str, max 140 chars), and 'keywords' (list[str], ~5 keywords), representing the miner's response.
    """

    # Required request input, filled by sending dendrite caller.
    query: typing.List[str]

    # Response: list of dicts with 'score', 'review', and 'keywords' keys.
    response: typing.Optional[typing.List[CheckerChainMinerResponse]] = None

    def deserialize(self) -> list[dict[str, typing.Any]]:
        """
        Deserialize the synapse response into a list of dictionaries.
        Each dictionary contains 'score', 'review', and 'keywords'.
        """
        if self.response is None:
            return []

        return [
            {
                "score": resp.score,
                "review": resp.review,
                "keywords": resp.keywords,
            }
            for resp in self.response
        ]
