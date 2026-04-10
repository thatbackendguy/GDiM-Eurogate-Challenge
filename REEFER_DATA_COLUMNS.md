# Reefer Data Columns

This file explains the columns in the reefer dataset provided to participants.

## Column descriptions

### `ContainerVisitID`

Unique identifier of the container visit at the terminal. One container can visit a terminal multiple times. Specific parameters are set on the container-visit level, and parameter values depend on the commodity.

### `ContainerIdentification`

Standardized container identification.

### `HardwareType`

Model of the reefer controller hardware used, for example `ML2` or `ML3`.

### `EventTime`

Timestamp representing the aggregated one-hour interval in UTC.

### `Power`

Average power consumption, measured in watts, during the hour.

### `Energy`

Total energy consumed, measured in watt-hours, during the hour.

### `EnergyTotal`

Total energy consumed, measured in watt-hours, since the start of the container visit.

### `TemperatureSetPoint`

Average set-point temperature, in degrees Celsius, defined for the reefer unit during the hour.

### `TemperatureAmbient`

Average ambient temperature outside the container, in degrees Celsius, measured during the hour.

### `TemperatureReturn`

Average temperature, in degrees Celsius, of air returning from the container cargo space during the hour.

### `TemperatureSupply`

Average temperature, in degrees Celsius, of air supplied into the cargo space during the hour.

### `LocationRack`

Most recent rack or slot location where the container was positioned.

### `ContainerSize`

Standardized container size/type according to ISO 6346.
