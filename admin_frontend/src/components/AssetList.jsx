import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  IconButton,
  Chip,
  Alert,
  CircularProgress,
  Box,
  Tooltip
} from '@mui/material';
import { Add, Edit, Delete, Factory } from '@mui/icons-material';
import apiClient from '../apiClient';
import { useLanguage } from '../contexts/LanguageContext';

function AssetList() {
  const { t } = useLanguage();
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [currentAsset, setCurrentAsset] = useState({ id: '', name: '', description: '' });

  const fetchAssets = async () => {
    setLoading(true);
    try {
      const data = await apiClient('/admin/assets');
      setAssets(data);
      setError(null);
    } catch (err) {
      setError(`${t.assets.fetchError}: ${err.message}`);
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
    setShowModal(true);
    setError(null);
  };

  const handleClose = () => {
    setShowModal(false);
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
      fetchAssets();
      handleClose();
    } catch (err) {
      setError(`${t.assets.saveError}: ${err.message}`);
    }
  };

  const handleDelete = async (assetId) => {
    if (window.confirm(t.assets.deleteConfirm.replace('{id}', assetId))) {
      try {
        await apiClient(`/admin/assets/${assetId}`, { method: 'DELETE' });
        fetchAssets();
      } catch (err) {
        alert(`${t.assets.deleteError}: ${err.message}`);
      }
    }
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '200px',
        color: '#4a5568',
        fontSize: '1.1rem'
      }}>
        ⏳ {t.assets.loading}
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '2rem',
        background: 'rgba(254, 178, 178, 0.9)',
        color: '#c53030',
        borderRadius: '12px',
        border: '1px solid #feb2b2',
        textAlign: 'center'
      }}>
        {error}
      </div>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4" component="h2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Factory color="primary" />
            {t.assets.title}
          </Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => handleOpen()}
            sx={{ borderRadius: '25px' }}
          >
            {t.assets.createNew}
          </Button>
        </Box>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>{t.assets.id}</TableCell>
                <TableCell>{t.assets.name}</TableCell>
                <TableCell>{t.assets.description}</TableCell>
                <TableCell>{t.assets.modelCount}</TableCell>
                <TableCell align="right">{t.assets.actions}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {assets.map((asset) => (
                <TableRow key={asset.id} hover>
                  <TableCell>
                    <Typography variant="body2" fontWeight="500">
                      {asset.id}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {asset.name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {asset.description || 'N/A'}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={asset.model_count}
                      size="small"
                      color={asset.model_count > 0 ? 'success' : 'default'}
                      variant={asset.model_count > 0 ? 'filled' : 'outlined'}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Box display="flex" gap={1} justifyContent="flex-end">
                      <Tooltip title="编辑资产">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleOpen(asset)}
                        >
                          <Edit />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title={asset.model_count > 0 ? "无法删除：存在关联模型" : "删除资产"}>
                        <span>
                          <IconButton
                            size="small"
                            color="error"
                            disabled={asset.model_count > 0}
                            onClick={() => handleDelete(asset.id)}
                          >
                            <Delete />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Material UI Dialog */}
        <Dialog 
          open={showModal} 
          onClose={handleClose}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>
            {isEditing ? t.assets.editAsset : t.assets.createAsset}
          </DialogTitle>
          <DialogContent>
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            
            <Box component="form" sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
              <TextField
                name="id"
                label={t.assets.assetId}
                value={currentAsset.id}
                onChange={handleChange}
                disabled={isEditing}
                placeholder="production_line_C"
                fullWidth
                helperText={isEditing ? t.assets.idCannotChange : ''}
              />
              
              <TextField
                name="name"
                label={t.assets.assetName}
                value={currentAsset.name}
                onChange={handleChange}
                fullWidth
              />
              
              <TextField
                name="description"
                label={t.assets.assetDescription}
                value={currentAsset.description}
                onChange={handleChange}
                multiline
                rows={4}
                fullWidth
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleClose} color="inherit">
              {t.assets.cancel}
            </Button>
            <Button onClick={handleSubmit} variant="contained">
              {isEditing ? t.assets.saveChanges : t.assets.create}
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
}

export default AssetList;