import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';
import {
    Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle,
    IconButton, Paper, Table, TableBody, TableCell, TableContainer, TableHead,
    TableRow, TextField, Typography, Alert
} from '@mui/material';
import { Edit, Delete } from '@mui/icons-material';

function AssetList() {
    const [assets, setAssets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [open, setOpen] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [currentAsset, setCurrentAsset] = useState({ id: '', name: '', description: '' });

    const fetchAssets = async () => {
        setLoading(true);
        try {
            const data = await apiClient('/admin/assets');
            setAssets(data);
        } catch (err) {
            setError(`Failed to fetch assets: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchAssets();
    }, []);

    const handleOpen = (asset = null) => {
        if (asset) {
            setIsEditing(true);
            setCurrentAsset(asset);
        } else {
            setIsEditing(false);
            setCurrentAsset({ id: '', name: '', description: '' });
        }
        setOpen(true);
        setError(null);
    };

    const handleClose = () => {
        setOpen(false);
    };

    const handleChange = (event) => {
        const { name, value } = event.target;
        setCurrentAsset(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async () => {
        try {
            if (isEditing) {
                await apiClient(`/admin/assets/${currentAsset.id}`, {
                    method: 'PUT',
                    body: JSON.stringify({ name: currentAsset.name, description: currentAsset.description }),
                });
            } else {
                await apiClient('/admin/assets', {
                    method: 'POST',
                    body: JSON.stringify(currentAsset),
                });
            }
            fetchAssets(); // Refresh list
            handleClose();
        } catch (err) {
            setError(`Failed to save asset: ${err.message}`);
        }
    };

    const handleDelete = async (assetId) => {
        if (window.confirm(`Are you sure you want to delete asset ${assetId}? This cannot be undone.`)) {
            try {
                await apiClient(`/admin/assets/${assetId}`, { method: 'DELETE' });
                fetchAssets(); // Refresh list
            } catch (err) {
                alert(`Failed to delete asset: ${err.message}`); // Use alert for immediate feedback on delete failure
            }
        }
    };

    if (loading && !assets.length) {
        return <CircularProgress />;
    }

    return (
        <Box sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4">Asset Management</Typography>
                <Button variant="contained" onClick={() => handleOpen()}>Create New Asset</Button>
            </Box>

            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>ID</TableCell>
                            <TableCell>Name</TableCell>
                            <TableCell>Description</TableCell>
                            <TableCell>Model Count</TableCell>
                            <TableCell align="right">Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {assets.map((asset) => (
                            <TableRow key={asset.id}>
                                <TableCell>{asset.id}</TableCell>
                                <TableCell>{asset.name}</TableCell>
                                <TableCell>{asset.description || 'N/A'}</TableCell>
                                <TableCell>{asset.model_count}</TableCell>
                                <TableCell align="right">
                                    <IconButton onClick={() => handleOpen(asset)}><Edit /></IconButton>
                                    <IconButton onClick={() => handleDelete(asset.id)} disabled={asset.model_count > 0}>
                                        <Delete />
                                    </IconButton>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>

            <Dialog open={open} onClose={handleClose}>
                <DialogTitle>{isEditing ? 'Edit Asset' : 'Create New Asset'}</DialogTitle>
                <DialogContent>
                    {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                    <TextField
                        autoFocus
                        margin="dense"
                        name="id"
                        label="Asset ID"
                        type="text"
                        fullWidth
                        variant="standard"
                        value={currentAsset.id}
                        onChange={handleChange}
                        disabled={isEditing}
                        helperText={isEditing ? "ID cannot be changed." : "A unique identifier, e.g., production_line_C"}
                    />
                    <TextField
                        margin="dense"
                        name="name"
                        label="Asset Name"
                        type="text"
                        fullWidth
                        variant="standard"
                        value={currentAsset.name}
                        onChange={handleChange}
                    />
                    <TextField
                        margin="dense"
                        name="description"
                        label="Description"
                        type="text"
                        fullWidth
                        multiline
                        rows={4}
                        variant="standard"
                        value={currentAsset.description}
                        onChange={handleChange}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleClose}>Cancel</Button>
                    <Button onClick={handleSubmit}>{isEditing ? 'Save Changes' : 'Create'}</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}

export default AssetList;
