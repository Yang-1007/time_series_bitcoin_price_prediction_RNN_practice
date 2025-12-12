#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """
    Baseline multi-step forecasting model using a single LSTM encoder.

    - Input:  (batch_size, sequence_length, num_features)
    - Output: (batch_size, prediction_length, num_features)

    Idea from lecture:
    The LSTM compresses recent history into a hidden state,
    which serves as a summary of short-term temporal dynamics.
    """

    def __init__(
        self,
        number_of_input_features: int,
        hidden_dimension: int,
        prediction_length: int,
    ) -> None:
        super().__init__()

        self.lstm_encoder = nn.LSTM(
            input_size=number_of_input_features,
            hidden_size=hidden_dimension,
            batch_first=True,
        )

        self.fully_connected_layer = nn.Linear(
            hidden_dimension,
            prediction_length * number_of_input_features,
        )

        self.prediction_length = prediction_length
        self.number_of_input_features = number_of_input_features

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        input_sequence shape: (batch_size, sequence_length, num_features)
        """
        _, (hidden_states, _) = self.lstm_encoder(input_sequence)

        # Use last layer's hidden state (robust to multi-layer LSTM)
        final_hidden_state = hidden_states[-1]  # (batch_size, hidden_dim)

        prediction_vector = self.fully_connected_layer(final_hidden_state)

        forecast = prediction_vector.view(
            input_sequence.size(0),
            self.prediction_length,
            self.number_of_input_features,
        )
        return forecast


class Seq2SeqLSTM(nn.Module):
    """
    Encoder–Decoder LSTM for multi-step forecasting.

    Lecture view:
    - Encoder summarizes past sequence
    - Decoder autoregressively generates future steps
    """

    def __init__(
        self,
        number_of_input_features: int,
        hidden_dimension: int,
        prediction_length: int,
    ) -> None:
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=number_of_input_features,
            hidden_size=hidden_dimension,
            batch_first=True,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=number_of_input_features,
            hidden_size=hidden_dimension,
            batch_first=True,
        )

        self.fully_connected_layer = nn.Linear(
            hidden_dimension,
            number_of_input_features,
        )

        self.prediction_length = prediction_length
        self.number_of_input_features = number_of_input_features

    def forward(
        self,
        input_sequence: torch.Tensor,
        target_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_sequence:  (batch_size, sequence_length, num_features)
        target_sequence: (batch_size, prediction_length, num_features) or None

        If target_sequence is provided → teacher forcing (training)
        If None → free-running decoding (validation / testing)
        """
        _, (hidden_state, cell_state) = self.encoder_lstm(input_sequence)

        decoder_input = input_sequence[:, -1:, :]
        forecast_outputs = []

        for t in range(self.prediction_length):
            decoder_output, (hidden_state, cell_state) = self.decoder_lstm(
                decoder_input,
                (hidden_state, cell_state),
            )

            prediction = self.fully_connected_layer(decoder_output)
            forecast_outputs.append(prediction)

            if target_sequence is not None:
                decoder_input = target_sequence[:, t:t + 1, :]
            else:
                decoder_input = prediction

        return torch.cat(forecast_outputs, dim=1)


class TransformerPredictor(nn.Module):
    """
    Transformer forecaster WITHOUT positional encoding.

    Used to demonstrate that attention alone, without positional encoding,
    cannot capture temporal ordering in time-series data.

    """

    def __init__(
        self,
        number_of_input_features: int,
        prediction_length: int,
        transformer_embedding_dimension: int = 32,
        number_of_attention_heads: int = 4,
        number_of_encoder_layers: int = 2,
    ) -> None:
        super().__init__()

        self.input_projection_layer = nn.Linear(
            number_of_input_features,
            transformer_embedding_dimension,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embedding_dimension,
            nhead=number_of_attention_heads,
            batch_first=False,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=number_of_encoder_layers,
        )

        self.fully_connected_layer = nn.Linear(
            transformer_embedding_dimension,
            prediction_length * number_of_input_features,
        )

        self.prediction_length = prediction_length
        self.number_of_input_features = number_of_input_features

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection_layer(input_sequence)
        embedded = embedded.permute(1, 0, 2)  # (T, B, d)

        encoded = self.transformer_encoder(embedded)
        final_embedding = encoded[-1]

        prediction_vector = self.fully_connected_layer(final_embedding)

        return prediction_vector.view(
            encoded.size(1),
            self.prediction_length,
            self.number_of_input_features,
        )


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds explicit temporal order information to Transformer inputs.
    """

    def __init__(
        self,
        embedding_dimension: int,
        maximum_sequence_length: int = 5000,
    ) -> None:
        super().__init__()

        position = torch.arange(maximum_sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2)
            * (-math.log(10000.0) / embedding_dimension)
        )

        positional_encoding = torch.zeros(maximum_sequence_length, embedding_dimension)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "positional_encoding",
            positional_encoding.unsqueeze(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (sequence_length, batch_size, embedding_dimension)
        Returns: same shape with positional information added

        """
        return x + self.positional_encoding[: x.size(0)]


class TransformerWithPositionalEncoding(nn.Module):
    """
    Transformer forecaster WITH positional encoding.

    Matches lecture explanation of why Transformers need
    explicit temporal information.
    """

    def __init__(
        self,
        number_of_input_features: int,
        prediction_length: int,
        transformer_embedding_dimension: int = 64,
        number_of_attention_heads: int = 4,
        number_of_encoder_layers: int = 2,
    ) -> None:
        super().__init__()

        self.input_projection_layer = nn.Linear(
            number_of_input_features,
            transformer_embedding_dimension,
        )

        self.positional_encoding = PositionalEncoding(
            embedding_dimension=transformer_embedding_dimension,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embedding_dimension,
            nhead=number_of_attention_heads,
            batch_first=False,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=number_of_encoder_layers,
        )

        self.fully_connected_layer = nn.Linear(
            transformer_embedding_dimension,
            prediction_length * number_of_input_features,
        )

        self.prediction_length = prediction_length
        self.number_of_input_features = number_of_input_features

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection_layer(input_sequence)
        embedded = embedded.permute(1, 0, 2)

        embedded = self.positional_encoding(embedded)
        encoded = self.transformer_encoder(embedded)

        final_embedding = encoded[-1]
        prediction_vector = self.fully_connected_layer(final_embedding)

        return prediction_vector.view(
            encoded.size(1),
            self.prediction_length,
            self.number_of_input_features,
        )

